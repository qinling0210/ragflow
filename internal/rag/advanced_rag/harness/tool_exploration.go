//
//  Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//

package harness

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"ragflow/internal/dao"
	"ragflow/internal/engine"
	"ragflow/internal/engine/types"
	"ragflow/internal/service"
	"ragflow/internal/service/nlp"
)

// Exploration providers (Go equivalent of Python harness/tools/exploration.py):
// knowledge-graph walks (graph_explore) and wiki page drill-downs (wiki_query),
// plus the open-web search provider seam.
//
// graph_explore walks the compiled knowledge graph (see ExploreGraph): it seeds
// entities by dense similarity, hops out over relations, then asks the model
// whether the subgraph answers the question. wiki_query searches the compiled
// wiki_page_draft rows. Together these mirror the exploration.py module, which
// hosts both graph_explore and wiki_query.
//
// Search seams stay independent of the search backend so the harness core is
// testable without one; the bridge wires concrete implementations (wiki_page_draft
// rows for SearchWiki, an HTTP client for WebSearcher).

// WebSearcher runs an open-web search and returns result strings. It is the Go
// equivalent of Python's web_search provider path wired into the action session.
type WebSearcher interface {
	Search(ctx context.Context, queries []string) ([]string, error)
}

// WikiPage is one compiled wiki page returned by WikiRetriever, mirroring the
// payload Python wiki_query emits per hit: the parsed page markdown plus the
// originating row identity so the harness can cite/load it.
type WikiPage struct {
	ChunkID string
	DocID   string
	DocName string
	Title   string
	// Content is the synthesised page markdown (Python's page_content).
	Content string
	// Score is the retrieval relevance (higher = better).
	Score float64
}

// WikiRetriever searches the compiled wiki_page_draft rows for a question and
// returns the most relevant pages. It is the Go equivalent of Python's
// tools/exploration.py::wiki_query, kept as a seam so the harness core stays
// independent of the search backend and remains testable without one.
type WikiRetriever interface {
	// SearchWiki returns up to topN compiled wiki pages for the question,
	// biased toward the optional keywords.
	SearchWiki(ctx context.Context, question string, keywords []string, topN int) ([]WikiPage, error)
}

// wikiQuery implements the `wiki_query` tool, mirroring Python
// tools/exploration.py::wiki_query + action_session._exec_wiki_query.
//
// It searches the compiled "wiki_page_draft" rows for the question, parses each
// hit's synthesised page markdown out of the row, and returns them in the same
// payload shape Python's action layer consumes: per hit
//
//	{chunk_id, doc_id, docnm_kwd, title, content, query}
//
// plus n_hits in Metrics. When no wiki retriever is wired (or the dataset has no
// wiki compilation), it returns a clean MISS so the model falls back to corpus
// tools — mirroring Python's wiki gate.
func (e *searchExecutor) wikiQuery(ctx context.Context, args map[string]any) (ToolOutcome, error) {
	if e.deps.WikiRetriever == nil {
		return ToolOutcome{
			Payload: []any{map[string]any{
				"kind": "wiki_query",
				"note": "wiki_query is not available for this dataset (no compiled wiki). Use corpus tools (retrieve/search_chunks) for in-document facts.",
			}},
			Status:  StatusMiss,
			Reason:  ReasonNoDoc,
			Metrics: map[string]any{},
		}, nil
	}

	// The wiki_query schema presents a 1-2 element array: [question, keywords?].
	queries := toolQueries(args)
	question := strings.TrimSpace(argString(args, "query"))
	keywords := toolStringList(args, "keywords")
	if question == "" && len(queries) > 0 {
		question = queries[0]
		if len(queries) > 1 {
			keywords = append(keywords, splitComma(queries[1])...)
		}
	}
	if question == "" {
		return ToolOutcome{Payload: []any{}, Status: StatusError, Reason: ReasonBadArgs}, nil
	}

	topN, _ := toInt(args["top_n"])
	if topN <= 0 {
		topN = 12 // Python _WIKI_QUERY_TOP_N = 12
	}
	if topN > 30 {
		topN = 30
	}

	pages, err := e.deps.WikiRetriever.SearchWiki(ctx, question, keywords, topN)
	if err != nil || len(pages) == 0 {
		return ToolOutcome{
			Payload: []any{map[string]any{
				"kind": "wiki_query",
				"note": "No compiled wiki page matched the question.",
			}},
			Status:  StatusMiss,
			Reason:  ReasonNoDoc,
			Metrics: map[string]any{},
		}, nil
	}

	passages := make([]any, 0, len(pages))
	for _, p := range pages {
		passages = append(passages, map[string]any{
			"chunk_id":  p.ChunkID,
			"doc_id":    p.DocID,
			"docnm_kwd": p.DocName,
			"title":     p.Title,
			"content":   p.Content,
			"query":     question,
		})
	}
	return ToolOutcome{
		Payload: passages,
		Status:  StatusOK,
		Reason:  ReasonNone,
		Metrics: map[string]any{"n_hits": len(passages)},
	}, nil
}

// splitComma splits a comma-separated keyword string into a clean slice,
// tolerating spaces. Mirrors Python's kw.strip() loop after split(",").
func splitComma(s string) []string {
	if s == "" {
		return nil
	}
	parts := strings.Split(s, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		if t := strings.TrimSpace(p); t != "" {
			out = append(out, t)
		}
	}
	return out
}

// graphExplore explores the compiled knowledge graph for a relational /
// multi-hop answer (ultra only).
//
// Delegates to ExploreGraph, which seeds entities for the query, hops along
// their relations, and returns either a direct answer or the source passages
// behind the relevant entities. Mirrors Python exploration.py::graph_explore,
// paired with wikiQuery above so the Go exploration.go matches the Python
// exploration.py module (which hosts both graph_explore and wiki_query).
func (e *searchExecutor) graphExplore(ctx context.Context, args map[string]any) (ToolOutcome, error) {
	query := argString(args, "query")
	if query == "" {
		query = e.req.Question
	}
	if query == "" {
		return ToolOutcome{Payload: []any{}, Status: StatusError, Reason: ReasonBadArgs}, nil
	}
	res, err := ExploreGraph(ctx, e.deps, e.deps.TenantID, e.deps.KbIDs, query, e.req.Keywords, toolDocScope(args))
	if err != nil {
		_LOG.Printf("[graph_explore] failed: %v", err)
		return ToolOutcome{Payload: []any{}, Status: StatusError, Reason: ReasonInfra}, nil
	}
	if len(res.Chunks) == 0 && res.Answer == "" {
		// No compiled knowledge graph for this dataset, or nothing matched — a
		// dataset-level dead end the ladder reports as EMPTY.
		return ToolOutcome{
			Payload: []any{map[string]any{
				"kind":  "graph_explore",
				"note":  "No compiled knowledge graph answered this query. Use search_chunks / navigate_structure instead.",
				"query": query,
			}},
			Status:  StatusEmpty,
			Reason:  ReasonNoStructure,
			Metrics: map[string]any{},
		}, nil
	}
	evidenceIDs := make([]string, 0, len(res.Chunks))
	if e.deps.KB != nil {
		// Merge the narrowed chunks together with their doc aggregations
		// (Python returns {"answer", "chunks", "doc_aggs"}).
		for _, idx := range e.deps.KB.Merge(res.Chunks, res.DocAggs) {
			evidenceIDs = append(evidenceIDs, fmt.Sprint(idx))
		}
	}
	passages := make([]any, 0, len(res.Chunks)+1)
	if res.Answer != "" {
		passages = append(passages, map[string]any{"kind": "graph_explore", "answer": res.Answer})
	}
	for _, c := range res.Chunks {
		passages = append(passages, passageFromChunk(c, query))
	}
	return ToolOutcome{
		Payload:     passages,
		EvidenceIDs: evidenceIDs,
		Status:      StatusOK,
		Reason:      ReasonNone,
		Metrics:     map[string]any{"chunks": len(res.Chunks)},
	}, nil
}

// ---------------------------------------------------------------------------
// graph_explore (knowledge-graph walk) — mirrors exploration.py::graph_explore
// ---------------------------------------------------------------------------

// ExploreGraph walks the compiled knowledge graph: seed entities for the query
// by dense similarity, hop kgHops out over their relations, then ask the model
// whether the resulting subgraph answers the question directly. When it does the
// answer is returned; when it doesn't, the source passages behind the relevant
// nodes are returned so the caller can keep researching. Mirrors Python
// exploration.py::graph_explore (which lives beside wiki_query in exploration.py).
//
// Seeds use the tenant embedding model via service.NavEmbedder (dense KNN,
// similarity >= kgSeedSim, re-ranked by mention_count_int desc); when the
// embedding model is unavailable it degrades to keyword match (mirrors Python
// _kg_search's `embed_mdl is None` path).
const (
	kgScopeDataset = "dataset"
	kgScopeDoc     = "doc"

	kgSeeds     = 2   // top-N entities matched directly to the question
	kgSeedPool  = 64  // KNN candidate pool before the mention_count_int re-sort
	kgSeedSim   = 0.8 // dense seed similarity floor (Python _KG_SEED_SIM)
	kgHops      = 2   // relation hops out from the seeds
	kgNeighbors = 128 // cap on neighbour entity rows resolved per hop
	kgRelLimit  = 32  // relations fetched per endpoint filter
)

type kgEntity struct {
	Name           string   `json:"name"`
	Type           string   `json:"type"`
	Description    string   `json:"description"`
	Aliases        []string `json:"aliases"`
	SourceChunkIDs []string `json:"source_chunk_ids"`
	DocID          string   `json:"doc_id"`
}

type kgRelation struct {
	From           string   `json:"from"`
	To             string   `json:"to"`
	Type           string   `json:"type"`
	SourceChunkIDs []string `json:"source_chunk_ids"`
	DocID          string   `json:"doc_id"`
}

// ExploreResult is the graph_explore output: exactly one of Answer / Chunks is
// populated, and DocAggs always reflects the returned set. Mirrors Python's
// {"answer", "chunks", "doc_aggs"} return dict.
type ExploreResult struct {
	Answer  string
	Chunks  []map[string]interface{}
	DocAggs []map[string]interface{}
}

// ExploreGraph implements graph_explore.
func ExploreGraph(ctx context.Context, deps SearchDeps, tenantID string, datasetIDs []string, query, keywords string, docScope []string) (ExploreResult, error) {
	empty := ExploreResult{}
	text := strings.TrimSpace(query + " " + keywords)
	if text == "" || len(datasetIDs) == 0 {
		return empty, nil
	}
	de := engine.Get()
	if de == nil {
		return empty, fmt.Errorf("graph_explore: engine not configured")
	}

	scopeKwd := kgScopeDataset
	if len(docScope) > 0 {
		scopeKwd = kgScopeDoc
	}

	var entities []kgEntity
	var relations []kgRelation
	seenNames := map[string]bool{}

	addEntities := func(new []kgEntity, scopeKey string) []string {
		var added []string
		for _, e := range new {
			key := scopeKey + ":" + strings.ToLower(e.Name)
			if seenNames[key] {
				continue
			}
			seenNames[key] = true
			entities = append(entities, e)
			added = append(added, e.Name)
		}
		return added
	}

	// Encode the seed text ONCE (not per dataset): a single embedding request
	// serves every KB. A nil vector means the embedding model is unavailable and
	// the seed search falls back to keyword match.
	seedVec := encodeSeedVector(ctx, deps, tenantID, text)

	for _, kbID := range datasetIDs {
		// (1) Seeds: dense KNN (similarity>=_KG_SEED_SIM) over the scoped entity
		// rows, re-ranked by mention_count_int desc; falls back to keyword match
		// when the embedding model is unavailable.
		seedRows := kgSeedSearch(ctx, de, tenantID, kbID, docScope, text, scopeKwd, seedVec)
		var seeds []kgEntity
		for _, r := range seedRows {
			if e, ok := kgParseEntity(r); ok {
				seeds = append(seeds, e)
			}
		}
		frontier := addEntities(seeds, kbID)

		// (2) Expand kgHops out, collecting relations and neighbour entities.
		for hop := 0; hop < kgHops; hop++ {
			if len(frontier) == 0 {
				break
			}
			terms := endpointTerms(frontier)
			relRows := kgSearch(ctx, de, tenantID, kbID, docScope, "relation", "", kgRelLimit, scopeKwd,
				map[string]interface{}{"from_entity_kwd": terms}, "", 0)
			relRows = append(relRows, kgSearch(ctx, de, tenantID, kbID, docScope, "relation", "", kgRelLimit, scopeKwd,
				map[string]interface{}{"to_entity_kwd": terms}, "", 0)...)
			seenRel := map[string]bool{}
			var hopRelations []kgRelation
			for _, r := range relRows {
				rel, ok := kgParseRelation(r)
				if !ok {
					continue
				}
				k := rel.From + "|" + rel.To + "|" + rel.Type
				if seenRel[k] {
					continue
				}
				seenRel[k] = true
				hopRelations = append(hopRelations, rel)
			}
			relations = append(relations, hopRelations...)

			// Neighbour entity names not yet visited.
			seenLower := map[string]bool{}
			for k := range seenNames {
				if strings.HasPrefix(k, kbID+":") {
					seenLower[strings.TrimPrefix(k, kbID+":")] = true
				}
			}
			neighSet := map[string]string{}
			for _, r := range hopRelations {
				for _, n := range []string{r.From, r.To} {
					n = strings.TrimSpace(n)
					if n == "" || seenLower[strings.ToLower(n)] {
						continue
					}
					neighSet[strings.ToLower(n)] = n
				}
			}
			if len(neighSet) == 0 {
				break
			}
			neighFiltered := make([]string, 0, len(neighSet))
			for _, n := range neighSet {
				neighFiltered = append(neighFiltered, n)
			}
			limit := kgNeighbors
			if len(neighFiltered) < limit {
				limit = len(neighFiltered)
			}
			neighRows := kgSearch(ctx, de, tenantID, kbID, docScope, "entity", "", limit, scopeKwd,
				map[string]interface{}{"name_kwd": endpointTerms(neighFiltered)}, "", 0)
			var neighbours []kgEntity
			for _, r := range neighRows {
				if e, ok := kgParseEntity(r); ok {
					neighbours = append(neighbours, e)
				}
			}
			frontier = addEntities(neighbours, kbID)
		}
	}

	if len(entities) == 0 && len(relations) == 0 {
		return empty, nil
	}

	// (3) Does the subgraph answer the question? The request-scoped model
	// (deps.Model, Python's tools.chat_mdl) is threaded through so the verdict is
	// drawn by the same tenant model as the rest of the harness.
	answer, relevant := askStructureAnswer(ctx, deps.Model, query, entities, relations)
	// (4a) Sufficient — return the answer, no chunks.
	if answer != "" {
		return ExploreResult{Answer: answer, DocAggs: []map[string]interface{}{}}, nil
	}

	// (4b) Insufficient — return source passages behind the relevant nodes,
	// narrowed to the sentences that carry the keywords (mirror Python
	// exploration.py L359-361). Python's _narrow_by_keywords keeps the whole set
	// when keywords is empty (text_processing.py `if not kwds: return chunks`),
	// so we gate the strict narrow on a non-empty keyword string to avoid
	// emptying the evidence. doc_aggs is computed from the resulting set (L363).
	evidence := collectEvidenceIDs(entities, relations, relevant)
	var chunks []map[string]interface{}
	for docID, ids := range evidence {
		if docID != "" && len(ids) > 0 {
			chunks = append(chunks, loadChunksByIDs(ctx, tenantID, ids)...)
		}
	}
	if strings.TrimSpace(keywords) != "" {
		chunks = NarrowByKeywords(chunks, keywords)
	}
	return ExploreResult{Chunks: chunks, DocAggs: DocAggs(chunks)}, nil
}

// seedEncoder encodes a seed text into a dense vector. It is a package seam so
// the seed search can be exercised without the tenant embedding service, which
// is backed by the database and therefore unavailable in unit tests.
//
// Returning nil is always valid: every caller falls back to keyword matching.
var seedEncoder = defaultSeedEncoder

// encodeSeedVector encodes the seed text once for the whole ExploreGraph call.
// Returns nil when the tenant embedding model is unavailable (or encoding fails).
//
// An embedder passed in via SearchDeps.Embedder (the external, Python-style
// "tools.embed_mdl" handle) is used when present; otherwise the call falls back
// to the package's internal resolver (defaultSeedEncoder), which needs a
// database and therefore stays self-contained for callers that cannot or do not
// want to supply one.
func encodeSeedVector(ctx context.Context, deps SearchDeps, tenantID, text string) []float64 {
	// Mirror Python's "if getattr(tools, 'embed_mdl', None)" guard: without a
	// wired database there is no tenant default embedding model, so skip encoding
	// and let the caller fall back to keyword matching rather than panicking.
	if dao.GetDB() == nil {
		return nil
	}
	if deps.Embedder != nil {
		// The external handle may itself panic on a misconfigured model service
		// (e.g. nil-DB); degrade to keyword matching like Python's
		// `except Exception` around get_vector, never crash the request.
		vecs, err := safeEncodeEmbedder(deps.Embedder, ctx, tenantID, text)
		if err != nil || len(vecs) == 0 || len(vecs[0]) == 0 {
			return nil
		}
		out := make([]float64, len(vecs[0]))
		for i, v := range vecs[0] {
			out[i] = float64(v)
		}
		return out
	}
	return seedEncoder(ctx, tenantID, text)
}

// safeEncodeEmbedder calls the embedder and recovers from any panic so a
// misconfigured embedding service degrades to keyword matching instead of
// crashing the request. It mirrors Python's get_vector exception handling.
func safeEncodeEmbedder(emb nlp.NavEmbedder, ctx context.Context, tenantID, text string) ([][]float32, error) {
	defer func() {
		if r := recover(); r != nil {
			_LOG.Printf("[graph_explore] seed encode panicked; falling back to keyword: %v", r)
		}
	}()
	return emb.Encode(ctx, tenantID, []string{text})
}

// SetSeedEncoder overrides the seed embedding used by graph exploration. It is
// a testing seam: the default encoder reaches the tenant embedding service,
// which needs a database that unit tests do not have. Pass nil to restore it.
// Returning nil is always valid — callers fall back to keyword matching.
func SetSeedEncoder(fn func(ctx context.Context, tenantID, text string) []float64) {
	if fn == nil {
		seedEncoder = defaultSeedEncoder
		return
	}
	seedEncoder = fn
}

func defaultSeedEncoder(ctx context.Context, tenantID, text string) []float64 {
	// Mirror Python's "if getattr(tools, 'embed_mdl', None)" guard: when no
	// database (and thus no tenant default embedding model) is wired up, skip
	// encoding entirely and fall back to keyword matching rather than panicking
	// on a nil DB inside the model-provider layer.
	if dao.GetDB() == nil {
		return nil
	}
	embedder := service.NewNavEmbedder(service.NewModelProviderService(), "")
	vecs, err := embedder.Encode(ctx, tenantID, []string{text})
	if err != nil || len(vecs) == 0 || len(vecs[0]) == 0 {
		return nil
	}
	vec := make([]float64, len(vecs[0]))
	for i, v := range vecs[0] {
		vec[i] = float64(v)
	}
	return vec
}

// kgSeedSearch searches the compiled KG entity rows for seeds (mirrors Python
// _kg_search dense branch): dense KNN over name_kwd with similarity>=0.8,
// re-ranked by mention_count_int desc, top kgSeeds. Falls back to keyword match
// when seedVec is nil (embedding model unavailable).
func kgSeedSearch(ctx context.Context, de engine.DocEngine, tenantID, kbID string, docIDs []string, text, scopeKwd string, seedVec []float64) []map[string]interface{} {
	if seedVec != nil {
		dense := &types.MatchDenseExpr{
			VectorColumnName:  fmt.Sprintf("q_%d_vec", len(seedVec)),
			EmbeddingData:     seedVec,
			EmbeddingDataType: "float",
			DistanceType:      "cosine",
			TopN:              kgSeedPool,
			ExtraOptions:      map[string]interface{}{"similarity": kgSeedSim},
		}
		rows := kgSearchRaw(ctx, de, tenantID, kbID, docIDs, "entity", scopeKwd, nil, []interface{}{dense}, "mention_count_int", kgSeedPool)
		return topMentionCount(rows, kgSeeds)
	}
	// Text fallback (mirrors Python _kg_search `embed_mdl is None` path).
	return kgSearch(ctx, de, tenantID, kbID, docIDs, "entity", text, kgSeeds, scopeKwd, nil, "mention_count_int", kgSeedPool)
}

// topMentionCount re-ranks rows by mention_count_int desc and returns topN.
func topMentionCount(rows []map[string]interface{}, topN int) []map[string]interface{} {
	sort.SliceStable(rows, func(i, j int) bool {
		return mentionCount(rows[i]) > mentionCount(rows[j])
	})
	if len(rows) > topN {
		rows = rows[:topN]
	}
	return rows
}

func mentionCount(row map[string]interface{}) int {
	switch v := row["mention_count_int"].(type) {
	case int:
		return v
	case int64:
		return int(v)
	case float64:
		return int(v)
	case float32:
		return int(v)
	}
	return 0
}

// kgSearchRaw is the low-level KG row search with explicit match exprs and any
// extra filter keys (e.g. from_entity_kwd/to_entity_kwd/name_kwd).
func kgSearchRaw(ctx context.Context, de engine.DocEngine, tenantID, kbID string, docIDs []string, kind, scopeKwd string, extra map[string]interface{}, matchExprs []interface{}, orderDesc string, limit int) []map[string]interface{} {
	idx := fmt.Sprintf("ragflow_%s", tenantID)
	condition := map[string]interface{}{"knowledge_graph_kwd": kind}
	if scopeKwd != "" {
		condition["scope_kwd"] = scopeKwd
	}
	if len(docIDs) > 0 {
		condition["doc_id"] = docIDs
	}
	for k, v := range extra {
		condition[k] = v
	}
	fields := []string{"content_with_weight", "source_chunk_ids", "doc_id", "docnm_kwd", "name_kwd", "mention_count_int", "from_entity_kwd", "to_entity_kwd"}
	req := &types.SearchRequest{
		IndexNames:   []string{idx},
		KbIDs:        []string{kbID},
		SelectFields: fields,
		Filter:       condition,
		Limit:        limit,
		MatchExprs:   matchExprs,
	}
	if orderDesc != "" {
		req.OrderBy = &types.OrderByExpr{}
		req.OrderBy.Desc(orderDesc)
	}
	res, err := de.Search(ctx, req)
	if err != nil {
		return nil
	}
	return res.Chunks
}

// kgSearch searches the compiled KG rows of one KB (mirrors Python _kg_search),
// using keyword match. It only builds the MatchTextExpr (with the pool-based
// TopN) and delegates the request construction to kgSearchRaw.
func kgSearch(ctx context.Context, de engine.DocEngine, tenantID, kbID string, docIDs []string, kind, text string, topN int, scopeKwd string, extra map[string]interface{}, orderDesc string, pool int) []map[string]interface{} {
	var matchExprs []interface{}
	if text != "" {
		knnTopN := topN
		if pool > knnTopN {
			knnTopN = pool
		}
		matchExprs = []interface{}{&types.MatchTextExpr{
			Fields:       []string{"content_ltks", "content_sm_ltks"},
			MatchingText: text,
			TopN:         knnTopN,
		}}
	}
	return kgSearchRaw(ctx, de, tenantID, kbID, docIDs, kind, scopeKwd, extra, matchExprs, orderDesc, topN)
}

func kgParseEntity(row map[string]interface{}) (kgEntity, bool) {
	name := ""
	payload := map[string]interface{}{}
	if s, ok := row["content_with_weight"].(string); ok {
		_ = json.Unmarshal([]byte(s), &payload)
	}
	if v, ok := payload["name"].(string); ok && v != "" {
		name = v
	} else if v, ok := payload["term"].(string); ok && v != "" {
		name = v
	} else if v, ok := payload["title"].(string); ok && v != "" {
		name = v
	}
	name = strings.TrimSpace(name)
	if name == "" {
		return kgEntity{}, false
	}
	e := kgEntity{
		Name:           name,
		Type:           strOr(payload["type"], "other"),
		Description:    strOr(payload["description"], ""),
		SourceChunkIDs: strSliceField(row["source_chunk_ids"]),
		DocID:          strOr(row["doc_id"], ""),
	}
	if aliases, ok := payload["aliases"].([]interface{}); ok {
		for _, a := range aliases {
			if s, ok := a.(string); ok && strings.TrimSpace(s) != "" {
				e.Aliases = append(e.Aliases, strings.TrimSpace(s))
			}
		}
	}
	return e, true
}

func kgParseRelation(row map[string]interface{}) (kgRelation, bool) {
	src := strings.TrimSpace(strOr(row["from_entity_kwd"], ""))
	tgt := strings.TrimSpace(strOr(row["to_entity_kwd"], ""))
	if src == "" || tgt == "" {
		return kgRelation{}, false
	}
	typ := "related"
	if payload, ok := row["content_with_weight"].(string); ok {
		var p map[string]interface{}
		if json.Unmarshal([]byte(payload), &p) == nil {
			if t, ok := p["type"].(string); ok && t != "" {
				typ = t
			} else if t, ok := p["relation"].(string); ok && t != "" {
				typ = t
			}
		}
	}
	return kgRelation{
		From: src, To: tgt, Type: typ,
		SourceChunkIDs: strSliceField(row["source_chunk_ids"]),
		DocID:          strOr(row["doc_id"], ""),
	}, true
}

// endpointTerms mirrors Python _endpoint_terms: original + lowercased forms, so
// hop queries match both merged (lowercased) and per-doc (original-case)
// endpoint fields.
func endpointTerms(names []string) []string {
	set := map[string]bool{}
	for _, n := range names {
		n = strings.TrimSpace(n)
		if n == "" {
			continue
		}
		set[n] = true
		set[strings.ToLower(n)] = true
	}
	out := make([]string, 0, len(set))
	for n := range set {
		out = append(out, n)
	}
	sort.Strings(out)
	return out
}

// collectEvidenceIDs mirrors Python _collect_evidence_ids: group source chunk ids
// of relevant entities AND relations by doc.
func collectEvidenceIDs(entities []kgEntity, relations []kgRelation, relevantNames []string) map[string][]string {
	wanted := map[string]bool{}
	for _, n := range relevantNames {
		if s := strings.TrimSpace(n); s != "" {
			wanted[strings.ToLower(s)] = true
		}
	}
	byDoc := map[string][]string{}
	seen := map[string]bool{}
	add := func(docID string, ids []string) {
		for _, cid := range ids {
			if cid == "" {
				continue
			}
			key := docID + "|" + cid
			if seen[key] {
				continue
			}
			seen[key] = true
			byDoc[docID] = append(byDoc[docID], cid)
		}
	}
	for _, e := range entities {
		names := map[string]bool{strings.ToLower(e.Name): true}
		for _, a := range e.Aliases {
			names[strings.ToLower(a)] = true
		}
		if intersects(names, wanted) {
			add(e.DocID, e.SourceChunkIDs)
		}
	}
	for _, r := range relations {
		if wanted[strings.ToLower(r.From)] || wanted[strings.ToLower(r.To)] {
			add(r.DocID, r.SourceChunkIDs)
		}
	}
	return byDoc
}

func intersects(a, b map[string]bool) bool {
	for k := range a {
		if b[k] {
			return true
		}
	}
	return false
}

// askStructureAnswer asks the chat model whether the subgraph answers the query
// (mirrors Python exploration.py::graph_explore calling _ask_structure),
// returning (answer, relevant_names). Delegates to AskStructure
// (structure_qa.go), the shared mirror of Python _ask_structure. `model` is the
// request-scoped chat model (Python's tools.chat_mdl); a nil model simply skips
// the verdict and yields no answer, matching Python's empty-verdict fallback.
func askStructureAnswer(ctx context.Context, model SessionModel, query string, entities []kgEntity, relations []kgRelation) (string, []string) {
	ems := make([]map[string]any, 0, len(entities))
	for _, e := range entities {
		ems = append(ems, map[string]any{
			"name": e.Name, "type": e.Type, "description": e.Description,
		})
	}
	rms := make([]map[string]any, 0, len(relations))
	for _, r := range relations {
		rms = append(rms, map[string]any{"from": r.From, "to": r.To, "type": r.Type})
	}
	return AskStructure(ctx, model, query, "knowledge graph", "Graph exploration", ems, rms)
}

func strOr(v interface{}, def string) string {
	if s, ok := v.(string); ok && s != "" {
		return s
	}
	return def
}

func strSliceField(v interface{}) []string {
	switch x := v.(type) {
	case []string:
		return x
	case []interface{}:
		var out []string
		for _, item := range x {
			if s, ok := item.(string); ok {
				out = append(out, s)
			}
		}
		return out
	}
	return nil
}
