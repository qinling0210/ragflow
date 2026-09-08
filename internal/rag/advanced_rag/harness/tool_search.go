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
	"log"
	"sort"
	"strings"
	"sync"

	"ragflow/internal/entity"
	"ragflow/internal/service/nlp"
)

// Search tools: the retrieval legs plus the narrowing that runs on top.
//
// Mirrors Python harness/tools/search.py::hybrid_search (with the vector /
// BM25 legs expressed as a weight on the same call, as the Python module does
// for its three entry points).

// Retrieval defaults. Mirrors Python search.py's _DEFAULT_* constants; callers
// that supply no configuration get exactly these values.
const (
	DefaultSimilarityThreshold   = 0.2
	DefaultTopN                  = 12
	DefaultRerankCandidatesCount = 64
	DefaultTopK                  = 1024
	// maxEffectiveQueryChars caps the expanded query (Python: [:400]).
	maxEffectiveQueryChars = 400

	// DefaultAgenticVectorWeight is the vector leg's weight when agentic
	// retrieval runs keyword-only: zero.
	//
	// This mirrors Python's RAGTools.retrieve, which takes
	// `using_embedding: bool = False` (agentic_rag.py:600) and where no caller
	// passes True, so `embd_mdl` stays None and the weight it forwards is 0.0
	// (agentic_rag.py:643-646). The agentic loop therefore runs on keyword
	// matches plus reranking; matching that is what keeps Go's recall aligned
	// with Python's. SearchDeps.UsingEmbedding is the Go spelling of that flag.
	DefaultAgenticVectorWeight = 0.0

	// DefaultHybridVectorWeight is the vector leg's weight when embedding IS
	// used (Python RAGTools.retrieve with using_embedding=True). Python reads it
	// from `_setting(self, "vector_similarity_weight", 0.7)` (agentic_rag.py:645);
	// 0.7 is the configured default when a caller turns embedding on.
	DefaultHybridVectorWeight = 0.7

	// HybridSearchDefaultVectorWeight is the vector leg's weight for the
	// standalone hybrid_search tool (Python search_chunks → hybrid_search). It
	// defaults to _DEFAULT_HYBRID_VECTOR_WEIGHT = 0.3 (search.py:51) — NOT 0.7 —
	// because the tool reads `_setting(tools, "vector_similarity_weight", 0.3)`.
	HybridSearchDefaultVectorWeight = 0.3
)

// Retriever is the retrieval backend the harness searches through.
//
// Mirrors Python's `settings.retriever.retrieval(...)`, which the harness calls
// as a module-level singleton. The Go version takes it as an interface so the
// session and the orchestrator can be tested without a search backend, and so
// the concrete adapter (the harness root package's RuntimeRetriever) stays the
// only place that knows about the runtime.
type Retriever interface {
	// Retrieve runs one search and returns raw chunk maps. The chunks carry at
	// least chunk_id / content / doc_id / docnm_kwd so the harness accessors
	// (ChunkIDOf, ChunkTextOf, DocIDOf, ...) can read them.
	Retrieve(ctx context.Context, req RetrieveRequest) ([]map[string]any, error)
}

// DocChunkLister returns a document's chunks in reading order, which is what
// the document-level tools need (Python settings.retriever.chunk_list with
// sort_by_position=True, rag/nlp/search.py:824).
//
// Retrieval cannot express it: RetrieveRequest ranks by relevance to a query,
// while these tools want every chunk of one document, page-ordered, paged.
// Like DocIDVerifier it is injected, so this package needs no document-store
// dependency of its own. Nil means the tools that need it stay unavailable.
type DocChunkLister interface {
	// DocChunks returns up to Limit chunks of DocID starting at Offset, ordered
	// by reading position. A short page means the document is exhausted.
	DocChunks(ctx context.Context, req DocChunksRequest) ([]map[string]any, error)
}

// DocChunksRequest is one page of one document's chunks.
type DocChunksRequest struct {
	DocID      string
	DatasetIDs []string
	TenantID   string
	Offset     int
	Limit      int
}

// DocIDVerifier reports which of a candidate document id set are known to the
// session's datasets. Mirrors Python RAGTools._filter_known_doc_ids, whose
// lookup is deliberately left to the caller so this package stays free of a
// database dependency.
type DocIDVerifier interface {
	// KnownDocIDs returns the subset of candidates that exist in the given
	// datasets. An error or an unavailable backend yields an empty set, which
	// the caller treats as "verification unavailable" rather than "no match".
	KnownDocIDs(ctx context.Context, datasetIDs, candidates []string) (map[string]bool, error)
}

// RetrieveRequest is one retrieval call.
//
// Weight is the vector-similarity weight: 0.3 hybrid (default), 1.0 vector-only,
// 0.0 keyword-only (BM25). This is how Python's three search entry points
// (hybrid_search / vector_search / bm25_search) differ.
type RetrieveRequest struct {
	Query                    string
	DatasetIDs               []string
	DocScope                 []string
	TopN                     int
	TopK                     int
	RerankCandidatesCount    int
	SimilarityThreshold      float64
	KeywordsSimilarityWeight float64
	TenantID                 string
	// MetaDataFilter restricts retrieval to chunks whose metadata matches
	// (Python tools.meta_data_filter). Nil means no filtering.
	MetaDataFilter map[string]any
	// RankFeature mirrors Python RAGTools.retrieve's `rank_feature` argument
	// (agentic_rag.py:668): question-type tags produced by
	// label_question(question, self.kbs) that the retriever uses to boost
	// matching chunks. The Go engine consumes it as a tag → weight map (matching
	// internal/engine/types and the chat pipeline), so it is map[string]float64,
	// not a bare list. Nil means no rank feature (Python passes None).
	RankFeature map[string]float64
	// ExcludeCompiled excludes compiled-product rows from plain retrieval
	// (Python hybrid_search passes must_not={"exists": "compile_kwd"},
	// search.py:171). Compiled products have their own expansion step, so the
	// base retrieval here should surface ordinary document chunks only. False
	// matches Python RAGTools.retrieve, which does not exclude them.
	ExcludeCompiled bool
}

// intOrDef returns v when set, else fallback.
func intOrDef(v, fallback int) int {
	if v > 0 {
		return v
	}
	return fallback
}

// floatOrDef returns v when set, else fallback. Kept separate from the pointer
// form below so both "unset" conventions stay explicit at the call site.
func floatOrDef(v, fallback float64) float64 {
	if v > 0 {
		return v
	}
	return fallback
}

// floatPtrOrDef returns *v when configured, else fallback. A pointer is used so
// a configured 0 is honoured instead of looking like "unset".
func floatPtrOrDef(v *float64, fallback float64) float64 {
	if v != nil {
		return *v
	}
	return fallback
}

// resolveVectorWeight mirrors the Python weight logic behind the vector leg.
// When embedding is off the weight is 0 (keyword-only); when on, the configured
// VectorSimilarityWeight wins, falling back to a default that depends on the
// caller:
//
//   - retrieve (HybridTool=false): Python RAGTools.retrieve uses_embedding
//     defaults the vector weight to 0.7 (agentic_rag.py:645) →
//     DefaultHybridVectorWeight.
//   - hybrid_search tool (HybridTool=true): Python search_chunks → hybrid_search
//     defaults to 0.3 (search.py:51 _DEFAULT_HYBRID_VECTOR_WEIGHT) →
//     HybridSearchDefaultVectorWeight.
func resolveVectorWeight(deps SearchDeps, hybridTool bool) float64 {
	if !deps.UsingEmbedding {
		return 0
	}
	def := DefaultHybridVectorWeight
	if hybridTool {
		def = HybridSearchDefaultVectorWeight
	}
	return floatPtrOrDef(deps.VectorSimilarityWeight, def)
}

// SearchDeps are the dependencies of HybridSearch.
type SearchDeps struct {
	// Backend is the retrieval service.
	Backend Retriever
	// KbIDs are the session's bound dataset ids (Python tools.kb_ids).
	KbIDs []string
	// TenantID scopes the retrieval.
	TenantID string
	// KB receives the raw chunks into the lossless memory store BEFORE
	// narrowing, and carries the per-request search cache.
	KB *Kbinfos
	// Expand runs the compiled-structure expansion when UseCompiled is set.
	// Nil disables compiled expansion.
	Expand CompiledExpander
	// NavRouter descends the dataset's compiled navigation tree for the
	// navigate_tree tool. Nil falls back to nav.NewNavServiceRouter() (the
	// internal/service/nav singleton).
	NavRouter NavTreeRouter
	// Model is the request-scoped chat model (Python tools.chat_mdl). It drives
	// the calculate tool's expression-writing call and the graph_explore
	// structure verdict (AskStructure). Required for `calculate`; nil makes the
	// model-backed tools report a miss.
	Model SessionModel
	// Logger is optional; nil uses the default logger.
	Logger *log.Logger
	// DocIDVerifier resolves which of the caller-supplied document ids actually
	// belong to the session's datasets (Python
	// RAGTools._filter_known_doc_ids). Nil means "no verification available":
	// the doc scope is passed through to the retriever unchanged, which is the
	// pre-existing behaviour.
	//
	// It is injected rather than queried here because resolving document
	// ownership is a database lookup this package deliberately does not depend
	// on; the caller (internal/rag/advanced_rag) supplies the implementation.
	DocIDVerifier DocIDVerifier
	// DocChunks pages through one document's chunks for the document-level
	// tools (fetch_full_document / summarize_document). Nil leaves both
	// unavailable.
	DocChunks DocChunkLister
	// Embedder is the external embedding handle used by graph/structure seed
	// encoding (Python tools.embed_mdl). Nil falls back to this package's
	// internal tenant-default resolver, which needs a database and therefore
	// stays self-contained for callers that cannot or do not want to supply one.
	Embedder nlp.NavEmbedder
	// Retrieval tuning (Python RAGTools.retrieve: _setting(self, "top_n"),
	// similarity_threshold, vector_similarity_weight, rerank_candidates_count,
	// top_k). Zero falls back to this package's Default* constants.
	TopN                int
	SimilarityThreshold float64
	// VectorSimilarityWeight is a pointer so an explicit 0 (keyword-only, the
	// agentic default) is distinguishable from "not configured", which falls
	// back to DefaultAgenticVectorWeight.
	VectorSimilarityWeight *float64
	// UsingEmbedding is the Go spelling of Python RAGTools.retrieve's
	// `using_embedding: bool = False` (agentic_rag.py:599). When false (the
	// agentic default), the vector leg is disabled and retrieval is keyword-only
	// (vector weight 0) — exactly Python's `embd_mdl = None; vector_weight = 0`.
	// When true, the embedder is engaged and the vector weight is applied:
	// VectorSimilarityWeight if set, else DefaultHybridVectorWeight (0.7, Python
	// `_setting(self, "vector_similarity_weight", 0.7)`). Unlike Python, the
	// query embedder is supplied by the runtime retrieval service, so Go carries
	// the flag rather than an embd_mdl handle on the retrieve call itself.
	UsingEmbedding        bool
	RerankCandidatesCount int
	TopK                  int
	// MetaDataFilter restricts retrieval to matching chunk metadata (Python
	// tools.meta_data_filter). Nil means no filtering.
	MetaDataFilter map[string]any
	// DocScope is the session-wide document restriction (Python
	// RAGTools.doc_scope). Empty means "search everything".
	DocScope []string
	// DoRefer mirrors Python RAGTools.do_refer: when true, summarize_document
	// prefixes the citation rules so the model cites the blocks it summarises.
	DoRefer bool
	// CiteRules is the optional user-defined citation template. Empty falls back
	// to the embedded citation_prompt.md (mirrors Python user_defined_prompts).
	// summarize_document passes it to the citation header verbatim.
	CiteRules string
	// WebSearch is the optional open-web provider for the `web_search` tool.
	// Nil means the tool is advertised (when the mode enables it) but reports a
	// clean MISS — mirroring Python's provider-gated web_search path.
	WebSearch WebSearcher
	// WikiRetriever is the optional provider for the `wiki_query` tool. Nil means
	// the tool is advertised (when HasWiki is set) but reports a clean MISS —
	// mirroring Python's wiki gate, where wiki_query only does real work when a
	// wiki_page_draft compilation exists for the dataset.
	WikiRetriever WikiRetriever
	// KBs mirrors Python RAGTools' self.kbs (agentic_rag.py:283) — the resolved
	// Knowledgebase objects (carrying parser_config / tenant_id) the agentic
	// tools run over. The rank feature is derived from these, exactly as
	// Python's retrieve calls label_question(question, self.kbs). Projected
	// from RAGTools.KBs when the search deps are built.
	KBs []*entity.Knowledgebase
	// Tagger mirrors Python RAGTools.retrieve's
	// `rank_feature=label_question(question, self.kbs)` (agentic_rag.py:668): it
	// classifies the query into question-type tags the retriever boosts on. Nil
	// means no tag boost (Python's label_question returning None). The Go
	// implementation lives in internal/service (MetadataService.LabelQuestion),
	// matching Python's rag.app.tag.label_question which queries the tag service.
	Tagger QuestionLabeler
}

// QuestionLabeler mirrors Python rag.app.tag.label_question(question, kbs)
// (agentic_rag.py:668 → tag.py). Given the query and the KB objects (which
// carry parser_config.tag_kb_ids), it returns a map of question-type tag →
// weight the retriever uses to rank results. The production implementation is
// internal/service.MetadataService.LabelQuestion; tests supply a stub.
type QuestionLabeler interface {
	LabelQuestion(ctx context.Context, question string, kbs []*entity.Knowledgebase) map[string]float64
}

// CompiledExpander enriches a search result with the dataset's compiled
// structure (page index / tree / knowledge graph / wiki synthesis pages).
//
// Mirrors Python tools/compiled_expansion.py::_expand_with_compiled, which
// hybrid_search calls when use_compiled is set. It is a seam rather than a
// direct call so the search leg stays independent of the compiled-structure
// machinery (and testable without a knowledge graph).
type CompiledExpander interface {
	Expand(ctx context.Context, kb *Kbinfos, query, keywords string, docScope []string) error
}

// HybridSearch mirrors Python tools/search.py::hybrid_search.
//
// Steps, in Python's order:
//  1. resolve limits and target datasets; bail when nothing is bound;
//  2. build the effective query (weighted query appended, capped at 400 chars);
//  3. serve an identical (query, datasets, topN, docScope) from the per-request
//     cache to avoid a duplicate round-trip;
//  4. retrieve;
//  5. stash the RAW chunks in the lossless memory store BEFORE narrowing;
//  6. narrow-or-keep by keywords (all-or-nothing, see narrowOrKeep);
//  7. optionally expand through the compiled structure;
//  8. cache and return.
func HybridSearch(ctx context.Context, deps SearchDeps, p SearchParams) ([]map[string]any, []map[string]any) {
	logger := deps.Logger
	if logger == nil {
		logger = _LOG
	}
	// Python retrieve:614-646 — an explicit argument wins, then the caller's
	// configuration, then this package's own defaults.
	topN := p.TopN
	if topN <= 0 {
		topN = deps.TopN
	}
	if topN <= 0 {
		topN = DefaultTopN
	}
	targetIDs := p.KbIDs
	if len(targetIDs) == 0 {
		targetIDs = deps.KbIDs
	}
	if len(targetIDs) == 0 {
		return nil, nil
	}
	if deps.Backend == nil {
		return nil, nil
	}
	docScope := resolveDocScope(ctx, deps, p.DocScope, targetIDs, logger)

	// 2. Effective query.
	var effectiveQuery string
	if strings.TrimSpace(p.RetrievalQuery) != "" {
		effectiveQuery = strings.TrimSpace(fmt.Sprintf("%s %s", p.Question, p.RetrievalQuery))
	} else if strings.TrimSpace(p.Keywords) != "" {
		effectiveQuery = strings.TrimSpace(fmt.Sprintf("%s %s", p.Question, p.Keywords))
	} else {
		effectiveQuery = p.Question
	}
	if len(effectiveQuery) > maxEffectiveQueryChars {
		effectiveQuery = effectiveQuery[:maxEffectiveQueryChars]
	}

	logger.Printf("[Hybrid search] Searching the knowledge base for %q (keywords: %s)", p.Question, p.Keywords)

	// 3. Per-request dedup: an identical query+scope is retrieved at most once,
	// so e.g. pre_search and a claim search asking the same question do not
	// repeat the round-trip, child fetch and narrowing.
	if deps.KB != nil {
		if chunks, aggs, ok := deps.KB.SearchCacheLoad(SearchCacheKey(effectiveQuery, targetIDs, topN, docScope)); ok {
			logger.Printf("[Hybrid search] Already searched this — reusing the %d passage(s) found earlier.", len(chunks))
			return chunks, aggs
		}
	}

	// 4. Retrieve.
	// rank_feature (Python retrieve: rank_feature=label_question(question,
	// self.kbs), agentic_rag.py:668). Go computes it from the KB objects via the
	// injected Tagger; nil Tagger ⇒ no boost (Python's label_question returns
	// None). The result is a tag → weight map the engine understands.
	var rankFeature map[string]float64
	if deps.Tagger != nil {
		rankFeature = deps.Tagger.LabelQuestion(ctx, effectiveQuery, deps.KBs)
	}
	chunks, err := deps.Backend.Retrieve(ctx, RetrieveRequest{
		Query:                 effectiveQuery,
		DatasetIDs:            targetIDs,
		DocScope:              docScope,
		TopN:                  topN,
		TopK:                  intOrDef(deps.TopK, DefaultTopK),
		RerankCandidatesCount: max(intOrDef(deps.RerankCandidatesCount, DefaultRerankCandidatesCount), topN),
		SimilarityThreshold:   floatOrDef(deps.SimilarityThreshold, DefaultSimilarityThreshold),
		// The field is named for keywords but carries the VECTOR weight, as in
		// Python's vector_similarity_weight. The default depends on the caller:
		// retrieve (HybridTool=false) uses 0.7 and the hybrid_search tool
		// (HybridTool=true) uses 0.3. ExcludeCompiled mirrors Python
		// hybrid_search's must_not={"exists":"compile_kwd"} (search.py:171),
		// which keeps compiled products out of plain retrieval.
		KeywordsSimilarityWeight: resolveVectorWeight(deps, p.HybridTool),
		TenantID:                 deps.TenantID,
		MetaDataFilter:           deps.MetaDataFilter,
		RankFeature:              rankFeature,
		ExcludeCompiled:          p.HybridTool,
	})
	if err != nil {
		logger.Printf("[Hybrid search] retrieval failed: %v", err)
		return nil, nil
	}

	// 5. doc_aggs from the FULL retrieved candidate set, computed here and left
	// untouched by the narrowing / compiled-expansion steps below — mirroring
	// Python, where `_normalize` takes doc_aggs straight from the retriever and
	// _narrow_or_keep / _expand_with_compiled only ever replace kbinfos["chunks"].
	aggs := DocAggs(chunks)

	// Memory BEFORE narrowing: the raw corpus may hold a fact the narrowing
	// drops, and a gap-driven grep over memory recovers it without re-querying.
	if deps.KB != nil {
		MemoryAdd(deps.KB, chunks)
	}

	// 6. Narrow-or-keep (chunks only; doc_aggs stays as retrieved).
	chunks = NarrowOrKeep(chunks, p.Keywords, "hybrid_search", logger)

	// 7. Compiled expansion.
	if p.UseCompiled && len(chunks) > 0 && deps.Expand != nil {
		logger.Printf("[Hybrid search] Compiled expansion enabled — enriching with page_index/tree/KG navigation.")
		if err := deps.Expand.Expand(ctx, deps.KB, p.Question, p.Keywords, docScope); err != nil {
			logger.Printf("[Hybrid search] compiled expansion failed: %v", err)
		}
	}

	if len(chunks) > 0 {
		logger.Printf("[Hybrid search] %q -> %d chunk(s): %s", trunc(p.Question, 80), len(chunks), docStatsLine(chunks))
	}

	// 8. Cache the result.
	if deps.KB != nil {
		deps.KB.SearchCacheStore(SearchCacheKey(effectiveQuery, targetIDs, topN, docScope), chunks, aggs)
	}
	return chunks, aggs
}

// SearchCacheKey mirrors Python _search_cache_key: the tuple of what actually
// determines a retrieval result. Scope and limits are part of the key so only a
// genuinely identical query is served from cache.
func SearchCacheKey(effectiveQuery string, targetIDs []string, topN int, docScope []string) string {
	query := normalizeSpace(effectiveQuery)
	ids := append([]string(nil), targetIDs...)
	sort.Strings(ids)
	docs := append([]string(nil), docScope...)
	sort.Strings(docs)
	return fmt.Sprintf("%s|%v|%d|%v", query, ids, topN, docs)
}

func normalizeSpace(s string) string {
	return strings.ToLower(strings.Join(strings.Fields(s), " "))
}

// DocAggs builds the per-document aggregation from a chunk list.
//
// Python receives these from the retriever itself (retrieval(..., aggs=True)),
// but the Go Retriever interface returns only chunks, so they are derived here.
// The shape stays compatible with the Python consumers: each entry carries
// doc_id (the key direct.py dedups on) plus the counts the reporting layer
// reads.
// docInDatasets reports whether docID belongs to the datasets being searched,
// mirroring the `kb_id.in_(self.kb_ids)` guard of Python
// RAGTools._resolve_doc_tenant (agentic_rag.py:987).
//
// The second result is false when ownership could not be determined — no
// verifier injected, or it failed. Callers then keep the previous behaviour
// instead of rejecting the document, so a lookup problem never turns into a
// silent "no such document".
func docInDatasets(ctx context.Context, deps SearchDeps, docID string) (belongs, verified bool) {
	if deps.DocIDVerifier == nil || docID == "" {
		return true, false
	}
	known, err := deps.DocIDVerifier.KnownDocIDs(ctx, deps.KbIDs, []string{docID})
	if err != nil || len(known) == 0 {
		return true, false
	}
	return known[docID], true
}

// resolveDocScope mirrors Python retrieve:623-635 — keep only the requested
// document ids that actually belong to the session's datasets.
//
// Two failure modes are deliberately non-destructive, because the document
// scope is supplied by model-generated tool arguments and must never let a
// lookup problem silently change what is searched:
//
//   - verification unavailable (no verifier injected, or it errored): the scope
//     is passed through unchanged;
//   - every candidate unknown: the scope is dropped and retrieval runs
//     unfiltered, matching Python's "falling back to unfiltered retrieval".
func resolveDocScope(ctx context.Context, deps SearchDeps, scope, kbIDs []string, logger *log.Logger) []string {
	if len(scope) == 0 || deps.DocIDVerifier == nil {
		return scope
	}
	candidates := make([]string, 0, len(scope))
	for _, d := range scope {
		if d != "" {
			candidates = append(candidates, d)
		}
	}
	if len(candidates) == 0 {
		return scope
	}
	known, err := deps.DocIDVerifier.KnownDocIDs(ctx, kbIDs, candidates)
	if err != nil || len(known) == 0 {
		logger.Printf("[Hybrid search] document-id verification unavailable (%d candidate(s)); searching the scope as given", len(candidates))
		return scope
	}
	valid := make([]string, 0, len(candidates))
	for _, d := range candidates {
		if known[d] {
			valid = append(valid, d)
		}
	}
	if len(valid) == 0 {
		logger.Printf("[Hybrid search] every supplied doc ID was unknown; falling back to unfiltered retrieval")
		return nil
	}
	return valid
}

func DocAggs(chunks []map[string]any) []map[string]any {
	if len(chunks) == 0 {
		return nil
	}
	type stat struct {
		count int
		chars int
		name  string
	}
	order := make([]string, 0, 8)
	stats := map[string]*stat{}
	for _, c := range chunks {
		if c == nil {
			continue
		}
		id := DocIDOf(c)
		if id == "" {
			id = DocTitleOf(c)
		}
		if id == "" {
			id = "?"
		}
		s, ok := stats[id]
		if !ok {
			s = &stat{name: DocTitleOf(c)}
			stats[id] = s
			order = append(order, id)
		}
		s.count++
		s.chars += len(ChunkTextOf(c))
	}
	out := make([]map[string]any, 0, len(order))
	for _, id := range order {
		s := stats[id]
		// Field names match runtime's referenceDocAggsFromRetrieval
		// (doc_id / doc_name / count) so these aggregations can be handed
		// straight to CanvasState.SetRetrievalReferences without translation.
		out = append(out, map[string]any{
			"doc_id":     id,
			"doc_name":   s.name,
			"count":      s.count,
			"char_count": s.chars,
		})
	}
	return out
}

// docStatsLine renders the per-document breakdown used by the search log line.
func docStatsLine(chunks []map[string]any) string {
	parts := make([]string, 0, 8)
	for _, a := range DocAggs(chunks) {
		parts = append(parts, fmt.Sprintf("%s:%vchunk(%vchars)",
			a["doc_id"], a["count"], a["char_count"]))
	}
	return strings.Join(parts, "; ")
}

// Normalisation of web-search payloads.
//
// Mirrors the shaping the Python harness applies to _exec_web_search results so
// they enter kbinfos with the same field names every other chunk carries.

// normalizeWebResults converts a raw web-search JSON payload into chunk maps.
// Accepts either {"chunks": [...]} or {"results": [...]}; entries without both a
// URL and content are dropped.
func normalizeWebResults(raw []byte) []map[string]any {
	var res struct {
		Chunks  []map[string]any `json:"chunks"`
		Results []map[string]any `json:"results"`
	}
	if err := json.Unmarshal(raw, &res); err != nil {
		return nil
	}
	src := res.Chunks
	if len(src) == 0 {
		src = res.Results
	}
	out := make([]map[string]any, 0, len(src))
	for i, c := range src {
		url := firstNonEmpty(anyString(c["url"]), anyString(c["link"]), anyString(c["source"]))
		if url == "" {
			continue
		}
		content := firstNonEmpty(anyString(c["content"]), anyString(c["raw_content"]), anyString(c["text"]))
		if content == "" {
			continue
		}
		docID := anyString(c["doc_id"])
		if docID == "" {
			docID = url
		}
		// doc_id stays the source URL; chunk_id must be UNIQUE per snippet so
		// Kbinfos.Merge (dedup by chunkKey) does not collapse several snippets
		// from the same URL, or the same URL across two retrieval rounds.
		chunkID := fmt.Sprintf("%s#%d", docID, i)
		out = append(out, map[string]any{
			"chunk_id":            chunkID,
			"content_with_weight": content,
			"doc_id":              docID,
			"docnm_kwd":           firstNonEmpty(anyString(c["title"]), anyString(c["source"])),
			"dataset_id":          anyString(c["dataset_id"]),
			"url":                 url,
			"source":              "web",
		})
	}
	return out
}

// firstNonEmpty returns the first argument that is not blank.
func firstNonEmpty(ss ...string) string {
	for _, s := range ss {
		if strings.TrimSpace(s) != "" {
			return s
		}
	}
	return ""
}

// ---------------------------------------------------------------------------
// Per-request retrieval cache (Python tools.search_cache).
//
// The Python side keeps a dict on the tools object keyed by the search
// parameters; an identical (query, scope, limits) search is served from cache
// instead of hitting the retriever again. Go attaches the same cache to
// Kbinfos and exposes it as methods here so the search tool and its cache live
// in one file, mirroring tools/search.py.
// ---------------------------------------------------------------------------

// searchCache is the per-request retrieval cache. Mirrors Python
// tools.search_cache (a dict on the tools object).
type searchCache struct {
	mu      sync.Mutex
	entries map[string]cachedSearch
}

// cachedSearch is one cache entry. Mirrors a Python tools.search_cache value.
type cachedSearch struct {
	chunks []map[string]any
	aggs   []map[string]any
}

// SearchCacheLoad returns a cached retrieval for key, or (nil, nil, false) when
// the key is absent. Mirrors Python `if cache_key in cache: return cache[cache_key]`.
func (k *Kbinfos) SearchCacheLoad(key string) ([]map[string]any, []map[string]any, bool) {
	c := k.ensureCache()
	c.mu.Lock()
	defer c.mu.Unlock()
	e, ok := c.entries[key]
	if !ok {
		return nil, nil, false
	}
	return e.chunks, e.aggs, true
}

// SearchCacheStore records a retrieval under key. Mirrors Python
// `cache[cache_key] = {...}`.
func (k *Kbinfos) SearchCacheStore(key string, chunks, aggs []map[string]any) {
	c := k.ensureCache()
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.entries == nil {
		c.entries = make(map[string]cachedSearch)
	}
	c.entries[key] = cachedSearch{chunks: chunks, aggs: aggs}
}

// ensureCache lazily initialises the per-request cache so a zero-value Kbinfos
// is usable without an explicit setup step.
func (k *Kbinfos) ensureCache() *searchCache {
	k.cacheOnce.Do(func() {
		if k.cache == nil {
			k.cache = &searchCache{}
		}
	})
	return k.cache
}
