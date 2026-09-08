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
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/cloudwego/eino/schema"

	"gorm.io/gorm"
	"ragflow/internal/agent/chat"
	"ragflow/internal/engine"
	"ragflow/internal/engine/types"
	"ragflow/internal/service/nav"
)

// Compiled-structure navigation: ontology_navigate / mindmap_navigate /
// navigate_tree / wiki_query / dataset-tree LLM routing.
//
// Mirrors Python harness/tools/navigation.py (the structure side of the module):
// structure navigation, the nav-tree router, the compiled-structure reader and
// dataset-tree LLM selection. The knowledge-graph walk (graph_explore) moved to
// exploration.go to match Python's exploration.py, which hosts both graph_explore
// and wiki_query; the shared seams (verdict struct, prompt, load helpers) stay
// here and are reached by those files through the harness package. In Go these
// were spread across several files (navigation / navtools / navservice /
// datasetnav / kg_explore); they are consolidated here.

// ---------------------------------------------------------------------------
// Structure navigation (ontology_navigate / mindmap_navigate)
// ---------------------------------------------------------------------------

var catalogKinds = map[string]bool{"tree": true, "timeline": true, "raptor": true, "page_index": true, "pageindex": true}
var mindmapKinds = map[string]bool{"mindmap": true, "mind_map": true}

const navSystemPrompt = `You are given the {noun} of one or more documents — an outline of entities and their relations — and a question.

Decide whether that outline alone already answers the question.

Rules:
1. Answer ONLY from the outline below. Do not invent facts.
2. Set "is_sufficient" to true only when the outline genuinely answers the question; otherwise false with an empty answer.
3. Always fill "relevant_entities" with the exact ` + "`name`" + ` values of the entities most related to the question (up to 10), even when the outline is not sufficient — they are used to pull the underlying source text.

Output ONLY JSON, no prose, no code fences:
{"is_sufficient": true/false, "answer": "<answer, or empty>", "relevant_entities": ["<entity name>", ...]}`

const (
	maxStructureEntities  = 300
	maxStructureRelations = 300
	maxEvidenceChunks     = 24
)

type structureEntity struct {
	Name           string    `json:"name"`
	Type           string    `json:"type"`
	Description    string    `json:"description"`
	SourceChunkIDs []string  `json:"source_chunk_ids"`
	DocID          string    `json:"-"`
	Vec            []float64 `json:"-"` // q_<dim>_vec row embedding, when the row carries one
}

type structureNavVerdict struct {
	IsSufficient     bool     `json:"is_sufficient"`
	Answer           string   `json:"answer"`
	RelevantEntities []string `json:"relevant_entities"`
}

// shapeKwds are the knowledge_graph_kwd ROW SHAPES a compiled structure can be
// written as. Reading BOTH the compact "graph" blob and the per-entity /
// per-relation rows is what mirrors Python _load_compiled_structure (which
// issues one query for the graph blob and a second for the per-entity rows and
// merges them). Otherwise navigation silently returns EMPTY for datasets that
// were compiled into per-entity/relation rows rather than a graph blob.
var shapeKwds = []string{"graph", "entity", "relation"}

func loadStructureEntities(ctx context.Context, tenantID, docID string, kinds map[string]bool, vecField string) []structureEntity {
	de := engine.Get()
	if de == nil {
		return nil
	}
	idx := fmt.Sprintf("ragflow_%s", tenantID)
	selectFields := []string{"content_with_weight", "compile_kwd", "compilation_template_kind_kwd", "knowledge_graph_kwd"}
	if vecField != "" {
		selectFields = append(selectFields, vecField)
	}
	req := &types.SearchRequest{
		IndexNames:   []string{idx},
		Filter:       map[string]interface{}{"doc_id": []string{docID}, "knowledge_graph_kwd": shapeKwds},
		SelectFields: selectFields,
		Limit:        3000,
	}
	res, err := de.Search(ctx, req)
	if err != nil {
		return nil
	}
	rows := make([]StructureRow, 0, len(res.Chunks))
	for _, row := range res.Chunks {
		kg, _ := row["knowledge_graph_kwd"].(string)
		sr := StructureRow{
			CompileKwd:        fmt.Sprint(row["compile_kwd"]),
			TemplateKind:      fmt.Sprint(row["compilation_template_kind_kwd"]),
			KnowledgeGraphKwd: kg,
			Content:           fmt.Sprint(row["content_with_weight"]),
		}
		if vecField != "" {
			if vec, ok := parseFloats(row[vecField]); ok {
				sr.Vec = vec
			}
		}
		rows = append(rows, sr)
	}
	kindList := make([]string, 0, len(kinds))
	for k := range kinds {
		kindList = append(kindList, k)
	}
	rawEntities, _ := ParseCompiledStructure(rows, kindList)
	var out []structureEntity
	for _, e := range rawEntities {
		name, _ := e["name"].(string)
		if strings.TrimSpace(name) == "" {
			continue
		}
		se := structureEntity{
			Name:        name,
			Type:        fmt.Sprint(e["type"]),
			Description: fmt.Sprint(e["description"]),
		}
		if v, ok := e["_vec"].([]float64); ok {
			se.Vec = v
		}
		if ids, ok := e["source_chunk_ids"].([]interface{}); ok {
			for _, id := range ids {
				if s, ok := id.(string); ok && s != "" {
					se.SourceChunkIDs = append(se.SourceChunkIDs, s)
				}
			}
		}
		out = append(out, se)
	}
	return out
}

// parseFloats normalizes an engine vector cell — a []float64, or a []any of
// numbers (engine decoders may return either) — into []float64.
func parseFloats(v any) ([]float64, bool) {
	switch t := v.(type) {
	case []float64:
		return t, true
	case []any:
		out := make([]float64, 0, len(t))
		for _, item := range t {
			f, ok := item.(float64)
			if !ok {
				return nil, false
			}
			out = append(out, f)
		}
		return out, true
	}
	return nil, false
}

func normalizeKind(row map[string]interface{}) string {
	if ck, _ := row["compile_kwd"].(string); ck == "raptor_graph" {
		return "raptor"
	}
	kind, _ := row["compilation_template_kind_kwd"].(string)
	if kind == "" {
		kind, _ = row["compile_kwd"].(string)
	}
	kind = strings.ToLower(strings.TrimSpace(strings.ReplaceAll(kind, "-", "_")))
	if kind == "pageindex" || kind == "page_index" || kind == "knowledge_graph" {
		return "timeline"
	}
	return kind
}

func loadChunksByIDs(ctx context.Context, tenantID string, ids []string) []map[string]interface{} {
	if len(ids) == 0 {
		return nil
	}
	de := engine.Get()
	if de == nil {
		return nil
	}
	idx := fmt.Sprintf("ragflow_%s", tenantID)
	limit := maxEvidenceChunks
	if len(ids) < limit {
		limit = len(ids)
	}
	req := &types.SearchRequest{
		IndexNames:   []string{idx},
		Filter:       map[string]interface{}{"id": ids},
		SelectFields: []string{"content_with_weight", "docnm_kwd", "doc_id"},
		Limit:        limit,
	}
	res, err := de.Search(ctx, req)
	if err != nil {
		return nil
	}
	var out []map[string]interface{}
	for _, row := range res.Chunks {
		out = append(out, map[string]interface{}{
			"chunk_id": row["id"], "content_with_weight": row["content_with_weight"],
			"docnm_kwd": row["docnm_kwd"], "doc_id": row["doc_id"],
		})
	}
	return out
}

func dedupStrings(in []string) []string {
	seen := map[string]bool{}
	var out []string
	for _, s := range in {
		if s != "" && !seen[s] {
			seen[s] = true
			out = append(out, s)
		}
	}
	return out
}

func orStr(v, def string) string {
	if v == "" {
		return def
	}
	return v
}

// ---------------------------------------------------------------------------
// The routing seam + NavResult payload
// ---------------------------------------------------------------------------

// Compiled-navigation tools: locate (dataset tree routing) and drill
// (in-document structure pinpointing).
//
// Mirrors Python harness/tools/navigation.py:
//   - _navigate_tree_impl      (line 830)
//   - _nav_search_titled       (line 659)
//   - NavResult                (line 62)
//   - _load_compiled_structure (line 111)
//   - _normalize_kind          (line 101)
//
// These are the payload builders; the tool dispatch lives in tool_executor.go.

// Navigation constants (Python navigation.py:655, 656, 773).
const (
	// navSearchMaxDocs is how many documents the hybrid nav search routes to.
	navSearchMaxDocs = 12
	// navMinDocScore drops documents below this score.
	navMinDocScore = 0.2
	// navTreeMaxDocs caps the documents surfaced by navigate_tree.
	navTreeMaxDocs = 8
)

// NavResult mirrors Python NavResult: the structured outcome of ONE
// compiled-navigation call.
//
// Text is what the MODEL sees (XML, unchanged). The remaining fields are the
// signals the ORCHESTRATOR routes on: does this dataset have the structure at
// all, did THIS query reach anything, and was the result worth using? Without
// them the caller can only regex the XML — which is how a `count="1"
// entities="0"` empty shell used to read as a successful hit.
type NavResult struct {
	// Text is the XML the model consumes.
	Text string
	// DocIDs are the documents reached (the routing result).
	DocIDs []string
	// RoutedDocs are (doc_id, summary) pairs — the routed documents WITH their
	// overall summaries. The summary is the "hint" fed back into retrieval: nav
	// is a hint, not a constraint, so these become a soft boost/rerank signal
	// instead of a hard doc_scope filter.
	RoutedDocs [][2]string
	// Entities are compiled entities discovered across those documents.
	Entities []map[string]any
	// Relations are compiled relations discovered across those documents.
	Relations []map[string]any
	// ChunkPaths maps chunk_id -> root→chunk structure path.
	ChunkPaths map[string]string
	// EmptyReason is "" when the call succeeded, else the machine-readable cause:
	// "infra" (no backend), "bad_args" (missing query), "no_structure"
	// (dataset-level: no compiled structure of this kind here), "no_doc"
	// (query-level: structure exists but THIS query reached nothing).
	EmptyReason string
}

// HasStructure reports whether the dataset has the compiled structure at all.
// A query-level miss is not a structure absence.
func (n NavResult) HasStructure() bool { return n.EmptyReason != ReasonNoStructure }

// navEmpty returns an empty NavResult with the given reason and matching status.
func navEmpty(reason, label string) NavResult {
	return NavResult{EmptyReason: reason}
}

// ---------------------------------------------------------------------------
// The routing seam
// ---------------------------------------------------------------------------

// NavTreeRouter descends a dataset's compiled navigation tree and returns the
// routed documents.
//
// Mirrors Python's `search_dataset_layers(kb.id, tenant_id, query,
// "navigation_tree", top_k, doc_scope)` — a hybrid BFS beam descent (vector +
// BM25) from the root clusters down to the nav_doc leaves.
//
// The Go side exposes this through internal/service/nav's NavService.Search,
// which is the same KNN-over-nav-rows capability. It is an interface here (not a
// direct call) for two reasons: the harness must not import the service layer,
// and a dataset without a compiled tree must be distinguishable from a query
// that routed to nothing.
type NavTreeRouter interface {
	// Route descends the nav tree for one dataset and returns the routed
	// documents with their summaries, ordered by descending score.
	// Returns (nil, nil) when the dataset has no compiled tree at all.
	Route(ctx context.Context, tenantID, kbID, query string, docScope []string, topK int) ([][2]string, error)
}

// NavTreeInput is one navigate_tree call.
type NavTreeInput struct {
	Query    string
	Keywords string
	DocScope []string
	// TenantID and KbIDs bound the datasets to route over.
	TenantID string
	KbIDs    []string
}

// NavigateTree mirrors Python _navigate_tree_impl: locate the document(s) most
// likely to hold the answer by descending the compiled navigation tree.
//
// This tool ROUTES, it does not retrieve. It deliberately does NOT fetch
// document content: loading full documents here cost ~79 ES queries per document
// only to render a snippet that navigate_structure supersedes — under per-pass
// fan-out that alone was enough to knock ES over.
//
// Each nav row already carries the document's overall summary, so labelling the
// route costs ZERO extra queries.
//
// There is deliberately NO chunk-retrieval fallback: when routing misses, the
// caller falls back to retrieve/search_chunks — the same work, owned by the
// orchestrator instead of hidden inside a "route" call.
func NavigateTree(ctx context.Context, router NavTreeRouter, in NavTreeInput) NavResult {
	query := strings.TrimSpace(in.Query)
	if query == "" {
		return navEmpty(ReasonBadArgs, "query is required")
	}
	if router == nil {
		return navEmpty(ReasonInfra, "no nav tree router")
	}
	// Descend on the topic + keywords: routing descends on similarity and does
	// not require keyword hits, so keywords only enrich the query.
	full := strings.TrimSpace(strings.TrimSpace(in.Query) + " " + strings.TrimSpace(in.Keywords))

	var (
		// ordered keeps the router's ranking: it returns documents in DESCENDING
		// score order, and that ordering IS the routing result. Re-sorting by
		// doc_id would discard the relevance the descent computed.
		ordered  [][2]string
		seen     = map[string]bool{}
		anyTree  bool
		anyError error
	)
	for _, kbID := range in.KbIDs {
		routed, err := router.Route(ctx, in.TenantID, kbID, full, in.DocScope, navSearchMaxDocs)
		if err != nil {
			anyError = err
			_LOG.Printf("[Dataset navigation search] nav-tree descent failed for kb=%s: %v", kbID, err)
			continue
		}
		if routed == nil {
			// No compiled tree for this dataset (distinguished from "routed to
			// nothing" by the router returning an empty non-nil slice).
			continue
		}
		anyTree = true
		for _, pair := range routed {
			did := strings.TrimSpace(pair[0])
			if did == "" || seen[did] {
				continue
			}
			seen[did] = true
			ordered = append(ordered, [2]string{did, strings.TrimSpace(pair[1])})
		}
	}
	if !anyTree {
		// Every dataset lacks a compiled tree — a DATASET-level fact, so the
		// caller may disable the tool for the session.
		return navEmpty(ReasonNoStructure, "no compiled navigation tree")
	}
	if len(ordered) == 0 {
		// Structure exists but THIS query reached nothing: a query-level miss.
		// The dataset may still have a tree a better-formed query would hit.
		return navEmpty(ReasonNoDoc, "no document routed")
	}
	_ = anyError

	// Cap after dedup, preserving the score order.
	if len(ordered) > navTreeMaxDocs {
		ordered = ordered[:navTreeMaxDocs]
	}
	docs := make([]string, 0, len(ordered))
	for _, pair := range ordered {
		docs = append(docs, pair[0])
	}

	var parts []string
	parts = append(parts, fmt.Sprintf(`<tree_navigation count="%d" query="%s">`, len(ordered), XMLEscape(query)))
	for i, pair := range ordered {
		if pair[1] != "" {
			parts = append(parts,
				fmt.Sprintf(`  <doc rank="%d" doc_id="%s">`, i+1, XMLEscape(pair[0])),
				fmt.Sprintf("    <summary>%s</summary>", XMLEscape(pair[1])),
				"  </doc>")
		} else {
			parts = append(parts, fmt.Sprintf(`  <doc rank="%d" doc_id="%s"/>`, i+1, XMLEscape(pair[0])))
		}
	}
	parts = append(parts, "</tree_navigation>")

	return NavResult{
		Text:       strings.Join(parts, "\n"),
		DocIDs:     docs,
		RoutedDocs: ordered,
	}
}

// ---------------------------------------------------------------------------
// Compiled-structure reading (in-document)
// ---------------------------------------------------------------------------

// StructureRow is one compiled-structure row, mirroring the doc-store fields
// Python reads (navigation.py:137-144).
type StructureRow struct {
	// CompileKwd distinguishes the COMPILE TYPE (tree / page_index / timeline /
	// raptor_graph / ...). NOT knowledge_graph_kwd.
	CompileKwd string
	// TemplateKind is the template-authored kind (compilation_template_kind_kwd).
	TemplateKind string
	// KnowledgeGraphKwd selects the ROW SHAPE:
	//   "graph"    → one compact blob whose content is {"entities":[], "relations":[]}
	//                (written by RAPTOR / tree compilation)
	//   "entity"   → one row per node, content is a single entity dict
	//   "relation" → one row per edge, content is a single relation dict
	//                (written by page_index and the pipeline Compiler tree)
	KnowledgeGraphKwd string
	// Content is the row's JSON payload.
	Content string
	// Vec is the row's q_<dim>_vec embedding, when the reader selected it. A
	// graph blob row carries the one vector its nested nodes share (RAPTOR); a
	// per-entity row carries that node's own vector (page_index).
	Vec []float64
}

// StructureReader reads a document's compiled structure rows.
//
// Mirrors Python _load_compiled_structure (navigation.py:111), which issues
// three doc-store queries (graph blob, per-entity/relation rows, raptor_graph)
// and merges the matching buckets.
//
// Go has no doc-store structured-query interface (see runtime.RetrievalService,
// which only exposes Search), so this is a seam: the harness ships the full
// parsing/merging logic below, and the reader is injected. Until a Go-side
// implementation is wired, navigate_structure reports EMPTY/no_structure and the
// navigation ladder falls through to `global` — the same behaviour as a dataset
// with no compiled structures.
type StructureReader interface {
	// ReadStructure returns the compiled rows for a document. Returns nil when
	// the backend has no compiled-structure support at all.
	ReadStructure(ctx context.Context, tenantID, kbID, docID string) ([]StructureRow, error)
}

// normalizeKind is defined above; rowKind projects a StructureRow into that
// shape. It mirrors Python _normalize_kind: the API's kind normalization
// (page_index / knowledge_graph → timeline), plus the raptor_graph override.
func rowKind(row StructureRow) string {
	return normalizeKind(map[string]any{
		"compile_kwd":                   row.CompileKwd,
		"compilation_template_kind_kwd": row.TemplateKind,
	})
}

// ParseCompiledStructure mirrors Python _load_compiled_structure's merge step
// (navigation.py:189-214): keep rows whose compile TYPE is in kinds, then split
// by row shape into entities / relations.
//
// Two row shapes coexist for compiled structures, and a given compile type may
// produce either:
//
//	graph blob    (knowledge_graph_kwd="graph"):    nested entities/relations
//	per-entity    (knowledge_graph_kwd="entity"):   one node per row
//	per-relation  (knowledge_graph_kwd="relation"): one edge per row
//
// Reading BOTH and merging is what makes navigation work regardless of which
// shape a compile type produced.
func ParseCompiledStructure(rows []StructureRow, kinds []string) ([]map[string]any, []map[string]any) {
	want := map[string]bool{}
	for _, k := range kinds {
		if k != "" {
			want[k] = true
		}
	}
	var entities, relations []map[string]any
	for _, row := range rows {
		kind := rowKind(row)
		if len(want) > 0 && !want[kind] {
			continue
		}
		graph, ok := decodeJSONObject(row.Content)
		if !ok {
			continue
		}
		attachVec := func(list []map[string]any) {
			if len(row.Vec) == 0 {
				return
			}
			for _, m := range list {
				m["_vec"] = row.Vec
			}
		}
		switch row.KnowledgeGraphKwd {
		case "graph":
			es := objectList(graph["entities"])
			rs := objectList(graph["relations"])
			attachVec(es)
			attachVec(rs)
			entities = append(entities, es...)
			relations = append(relations, rs...)
		case "entity":
			attachVec([]map[string]any{graph})
			entities = append(entities, graph)
		case "relation":
			attachVec([]map[string]any{graph})
			relations = append(relations, graph)
		}
	}
	return entities, relations
}

func decodeJSONObject(s string) (map[string]any, bool) {
	v := ExtractJSON(s)
	m, ok := v.(map[string]any)
	return m, ok
}

func objectList(v any) []map[string]any {
	list, _ := v.([]any)
	out := make([]map[string]any, 0, len(list))
	for _, item := range list {
		if m, ok := item.(map[string]any); ok {
			out = append(out, m)
		}
	}
	return out
}

// ---------------------------------------------------------------------------
// Dataset-tree LLM selection (P1b)
// ---------------------------------------------------------------------------

// Dataset-nav router tunables mirror Python navigation.py.
const (
	navMaxDocs          = 8
	navMaxClusters      = 500
	navChildrenPageSize = 1000
	navTreeMaxDepth     = 6
	navTreeMaxLeaves    = 300
)

// navSelectSystem mirrors Python _NAV_SELECT_SYSTEM.
const navSelectSystem = `You are routing a question through a dataset's navigation tree.

You are given a QUESTION and a numbered list of {noun}, each with a name and a short description.
Choose the {noun} most likely to contain information relevant to answering the question.

Rules:
1. Judge only from the names and descriptions shown.
2. Be selective — include an item only if it is plausibly relevant. Include several when several are equally plausible.
3. If none are clearly relevant, return an empty list.
4. Return the bracketed index numbers of the chosen {noun}.

Output ONLY JSON, no prose, no code fences:
{"relevant": [<index>, ...]}`

type navSelectVerdict struct {
	Relevant []int `json:"relevant"`
}

// NavigateDatasetByTree walks the dataset nav tree with two LLM passes
// (cluster-select → document-select) and returns the routed doc_ids (capped at
// navMaxDocs). This is the LLM two-round selection (P1b) implemented in the
// harness package, which can import the chat invoker (agent/tool cannot without
// an import cycle). It routes only — it does not retrieve.
func NavigateDatasetByTree(ctx context.Context, db *gorm.DB, ns nav.NavService, tenantID, kbID, query string) []string {
	query = strings.TrimSpace(query)
	if query == "" {
		return nil
	}

	// 1. List top-level clusters.
	clusters, _, err := ns.ListClusters(ctx, tenantID, kbID, 0, navMaxClusters)
	if err != nil {
		_LOG.Printf("datasetnav: list clusters failed: %v", err)
		return nil
	}
	if len(clusters) == 0 {
		return nil
	}
	clusterItems := make([]navSelectItem, len(clusters))
	for i, c := range clusters {
		clusterItems[i] = navSelectItem{Name: c.Name, Description: c.Description, DocCount: c.DocCount}
	}

	// 2. LLM selects relevant clusters.
	selected := askNavSelect(ctx, db, query, "clusters", clusterItems, navMaxClusters)
	if len(selected) == 0 {
		return nil
	}

	// 3. BFS-descend selected clusters to document leaves.
	leaves := collectNavLeaves(ctx, ns, tenantID, kbID, selected)

	// 4. LLM selects relevant documents.
	docs := askNavSelect(ctx, db, query, "documents", leaves, navTreeMaxLeaves)

	// Dedup + cap.
	seen := map[string]bool{}
	var routed []string
	for _, d := range docs {
		if d.DocID == "" || seen[d.DocID] {
			continue
		}
		seen[d.DocID] = true
		routed = append(routed, d.DocID)
		if len(routed) >= navMaxDocs {
			break
		}
	}
	return routed
}

// navSelectItem is a renderable cluster/document with a name + description.
type navSelectItem struct {
	Name        string
	Description string
	DocCount    int
	DocID       string
}

// askNavSelect renders items as a numbered list and asks the model which indices
// are relevant. Returns the selected items (a subset). Mirrors _ask_nav_select.
func askNavSelect(ctx context.Context, db *gorm.DB, query, noun string, items []navSelectItem, maxItems int) []navSelectItem {
	if len(items) == 0 {
		return nil
	}
	capped := items
	if len(capped) > maxItems {
		capped = capped[:maxItems]
	}
	var b strings.Builder
	for i, it := range capped {
		name := strings.TrimSpace(it.Name)
		if name == "" {
			name = fmt.Sprintf("item-%d", i)
		}
		desc := strings.Join(strings.Fields(it.Description), " ")
		if len(desc) > 300 {
			desc = desc[:300]
		}
		b.WriteString(fmt.Sprintf("[%d] %s", i, name))
		if it.DocCount > 0 {
			b.WriteString(fmt.Sprintf(" [%d docs]", it.DocCount))
		}
		b.WriteString(": " + desc + "\n")
	}

	system := strings.ReplaceAll(navSelectSystem, "{noun}", noun)
	user := fmt.Sprintf("Question:\n%s\n\n%s (numbered):\n%s\n\nOutput JSON:", query, strings.Title(noun), b.String())

	inv := chat.GetDefaultInvoker()
	if inv == nil {
		_LOG.Printf("datasetnav: LLM %s selection skipped (chat invoker not configured)", noun)
		return nil
	}
	resp, err := inv.Invoke(ctx, db, chat.Request{
		Messages: []schema.Message{
			{Role: schema.System, Content: system},
			{Role: schema.User, Content: user},
		},
	})
	if err != nil {
		_LOG.Printf("datasetnav: LLM %s selection failed: %v", noun, err)
		return nil
	}
	var v navSelectVerdict
	if err := UnmarshalModelJSON(resp.Content, &v); err != nil {
		return nil
	}
	seen := map[int]bool{}
	var out []navSelectItem
	for _, idx := range v.Relevant {
		if idx < 0 || idx >= len(capped) || seen[idx] {
			continue
		}
		seen[idx] = true
		out = append(out, capped[idx])
	}
	return out
}

// collectNavLeaves BFS-descents selected clusters to document leaves. Mirrors
// Python _collect_nav_leaves.
func collectNavLeaves(ctx context.Context, ns nav.NavService, tenantID, kbID string, selected []navSelectItem) []navSelectItem {
	type node struct {
		name  string
		depth int
	}
	frontier := make([]node, 0, len(selected))
	for _, c := range selected {
		if c.Name != "" {
			frontier = append(frontier, node{c.Name, 0})
		}
	}
	var leaves []navSelectItem
	seenDocs := map[string]bool{}
	seenNodes := map[string]bool{}
	for len(frontier) > 0 && len(leaves) < navTreeMaxLeaves {
		cur := frontier[0]
		frontier = frontier[1:]
		if seenNodes[cur.name] {
			continue
		}
		seenNodes[cur.name] = true
		children, _, err := ns.ListChildren(ctx, tenantID, kbID, cur.name, 0, navChildrenPageSize)
		if err != nil {
			continue
		}
		for _, ch := range children {
			if ch.Type == "doc" {
				did := strings.TrimSpace(ch.DocID)
				if did != "" && !seenDocs[did] {
					seenDocs[did] = true
					leaves = append(leaves, navSelectItem{Name: ch.Name, Description: ch.Description, DocID: did})
					if len(leaves) >= navTreeMaxLeaves {
						break
					}
				}
			} else if ch.Type == "cluster" && ch.Name != "" && cur.depth+1 < navTreeMaxDepth {
				frontier = append(frontier, node{ch.Name, cur.depth + 1})
			}
		}
	}
	return leaves
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// In-document structure drill-down (mirrors navigation.py, distributed helpers)
// ---------------------------------------------------------------------------
//
// These are the Go counterparts of the functions navigation.py DEFINES in its
// own body. Like Python, they reuse primitives imported from elsewhere: cosine /
// appendUnique (compiled_expansion.go, action_session.go), XMLEscape / Snippet /
// ChunkTextOf (chunk_utils.go). They stay here so navigation.go owns the
// navigate-tree / navigate-structure algorithm, mirroring how navigation.py owns
// its orchestration while importing its primitives.

// Structure drill-down bounds (mirror navigation.py's module-level limits).
const (
	structMaxDepth     = 12                             // _STRUCT_MAX_DEPTH: stop walking past this depth
	structMaxNodes     = 300                            // _STRUCT_MAX_NODES: cap on nodes kept per level
	structBranchK      = 12                             // _STRUCT_BRANCH_K: children kept per node
	structVecBeamRatio = 0.65                           // _STRUCT_VEC_BEAM_RATIO: score floor relative to best
	structRelevanceMin = 1                              // min keyword relevance to keep a node without a vector
	structDescSnippet  = 300                            // snippet length for a node's description in the outline
	structMaxChunks    = 6                              // _STRUCT_MAX_CHUNKS: top chunks exposed per drilled doc
	structCatalogKinds = "tree_node|page_index|section" // kinds eligible for catalog outline

	// structTocMaxDepth bounds the ancestor walk in renderTocDrilldown's
	// chunk-retrieval and llm_toc branches (Python _STRUCT_TOC_MAX_DEPTH). The
	// remaining _STRUCT_TOC_* limits gate the whole-TOC LLM pass, which Python
	// leaves off by default; they are not mirrored (they would be unreachable in
	// Go).
	structTocMaxDepth = 6

	// Chunk-index selection bounds for structures that cannot score themselves
	// (RAPTOR blob path). Mirror _STRUCT_RECALL_TOP_N and _STRUCT_MAX_CHUNK_HITS.
	structRecallTopN   = 24 // _STRUCT_RECALL_TOP_N: hybrid chunk hits recalled per document
	structMaxChunkHits = 8  // _STRUCT_MAX_CHUNK_HITS: retrieved chunks shown per drilled doc
)

// queryTerms mirrors navigation.py's regex tokenizer: lowercase coarse word
// tokens of length >= 2 (the BM25-ish keyword signal).
func queryTerms(q string) []string {
	if q == "" {
		return nil
	}
	lower := strings.ToLower(q)
	var out []string
	var cur []byte
	for i := 0; i < len(lower); i++ {
		c := lower[i]
		if (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' {
			cur = append(cur, c)
			continue
		}
		if len(cur) >= 2 {
			out = append(out, string(cur))
		}
		cur = cur[:0]
	}
	if len(cur) >= 2 {
		out = append(out, string(cur))
	}
	return out
}

// nodeRelevance mirrors navigation.py _node_relevance: how many query terms occur
// in a TOC node's name+description.
func nodeRelevance(terms []string, name, desc string) int {
	if len(terms) == 0 {
		return 0
	}
	text := strings.ToLower(name + " " + desc)
	n := 0
	for _, t := range terms {
		if strings.Contains(text, t) {
			n++
		}
	}
	return n
}

// structureNode is a node of a compiled structure hierarchy during drill-down:
// the entity plus its optional embedding vector (for cosine scoring).
type structureNode struct {
	name           string
	nodeType       string
	desc           string
	sourceChunkIDs []string
	vec            []float64
}

// structureRel is a parent->child relation of a compiled structure hierarchy.
type structureRel struct {
	from    string
	to      string
	relType string
}

// buildTocTree mirrors navigation.py _build_toc_tree: a name->node map, the
// parent->children map, the child->parent map (single parent, last-wins) and the
// root names (tree nodes with no parent; isolated nodes are their own roots).
func buildTocTree(nodes []structureNode, rels []structureRel) (map[string]structureNode, map[string][]string, map[string]string, []string) {
	byName := map[string]structureNode{}
	for _, n := range nodes {
		if n.name != "" {
			byName[n.name] = n
		}
	}
	children := map[string][]string{}
	parents := map[string]string{}
	for _, r := range rels {
		p, c := r.from, r.to
		if p == "" || c == "" || p == c {
			continue
		}
		if _, ok := children[p]; !ok {
			children[p] = []string{}
		}
		dup := false
		for _, x := range children[p] {
			if x == c {
				dup = true
				break
			}
		}
		if !dup {
			children[p] = append(children[p], c)
		}
		parents[c] = p
	}
	var roots []string
	for n := range byName {
		if _, has := parents[n]; !has {
			roots = append(roots, n)
		}
	}
	if len(roots) == 0 {
		for n := range byName {
			if _, has := children[n]; !has {
				roots = append(roots, n)
			}
		}
	}
	if len(roots) == 0 {
		for n := range byName {
			roots = append(roots, n)
		}
	}
	return byName, children, parents, roots
}

// chunkPtrs mirrors navigation.py _chunk_ptrs: a bounded, deduped comma-joined
// slice of an item's source chunk ids (<=8, the anchors the model sees).
func chunkPtrs(ids []string) string {
	if len(ids) == 0 {
		return ""
	}
	var seen []string
	for _, c := range ids {
		if c == "" {
			continue
		}
		if !sliceContains(seen, c) {
			seen = append(seen, c)
			if len(seen) >= 8 {
				break
			}
		}
	}
	return strings.Join(seen, ",")
}

func sliceContains(s []string, v string) bool {
	for _, x := range s {
		if x == v {
			return true
		}
	}
	return false
}

// collectChunkIDs mirrors navigation.py _collect_chunk_ids: a deduped, capped
// union of source chunk ids across a set of nodes.
func collectChunkIDs(nodes []structureNode, capN int) []string {
	if capN <= 0 {
		capN = 32
	}
	var out []string
	for _, n := range nodes {
		for _, c := range n.sourceChunkIDs {
			if c == "" {
				continue
			}
			if sliceContains(out, c) {
				continue
			}
			out = append(out, c)
			if len(out) >= capN {
				return out
			}
		}
	}
	return out
}

// nodeScore mirrors navigation.py _node_score: cosine when a query vector and the
// node embedding both exist, else keyword relevance.
func nodeScore(qvec []float64, terms []string, n structureNode) float64 {
	if len(qvec) > 0 && len(n.vec) > 0 {
		return cosine(qvec, n.vec)
	}
	return float64(nodeRelevance(terms, n.name, n.desc))
}

// vecsEqual reports whether two node embeddings are element-wise equal, mirroring
// navigation.py _vecs_equal. Zero-length or differently-sized vectors are unequal.
func vecsEqual(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// hasDistinctNodeVectors reports whether the nodes carry per-node embeddings
// rather than one shared vector. Mirrors navigation.py _has_distinct_node_vectors:
// a page_index / per-node tree has distinct node vectors, while a RAPTOR blob
// inherits the blob's one vector for every node. Returns false when fewer than
// two nodes carry a vector or all share the same one.
func hasDistinctNodeVectors(nodes []structureNode) bool {
	var first []float64
	haveFirst := false
	for _, n := range nodes {
		if len(n.vec) == 0 {
			continue
		}
		if !haveFirst {
			first = n.vec
			haveFirst = true
			continue
		}
		if !vecsEqual(first, n.vec) {
			return true
		}
	}
	return false
}

// nodesCoveringChunks returns the names of nodes whose source_chunk_ids cover any
// of chunkIDs, mirroring navigation.py _nodes_covering_chunks. Used only to give
// retrieved chunks a structural home; a chunk no node covers is still shown.
func nodesCoveringChunks(nodes []structureNode, chunkIDs []string) []string {
	if len(chunkIDs) == 0 {
		return nil
	}
	want := make(map[string]bool, len(chunkIDs))
	for _, c := range chunkIDs {
		if c != "" {
			want[c] = true
		}
	}
	var names []string
	for _, n := range nodes {
		name := strings.TrimSpace(n.name)
		if name == "" {
			continue
		}
		for _, c := range n.sourceChunkIDs {
			if c != "" && want[c] {
				names = append(names, name)
				break
			}
		}
	}
	return names
}

// recallChunkIDsInDoc mirrors navigation.py _recall_chunk_ids_in_doc: for a
// structure whose nodes share one blob vector (RAPTOR) it asks the retrieval
// backend for the document's most relevant ORIGINAL chunks (the backend already
// excludes compiled rows), and returns them as (id, score) hits. A nil
// retriever, empty query, or an error yields no hits.
func recallChunkIDsInDoc(ctx context.Context, retriever Retriever, query, docID, tenantID string, datasetIDs []string, topN int) []chunkHit {
	if retriever == nil || query == "" || docID == "" {
		return nil
	}
	if topN <= 0 {
		topN = structRecallTopN
	}
	chunks, err := retriever.Retrieve(ctx, RetrieveRequest{
		Query:      query,
		DatasetIDs: datasetIDs,
		DocScope:   []string{docID},
		TopN:       topN,
		TenantID:   tenantID,
	})
	if err != nil || len(chunks) == 0 {
		return nil
	}
	hits := make([]chunkHit, 0, len(chunks))
	for _, c := range chunks {
		id, _ := c["chunk_id"].(string)
		if id == "" {
			continue
		}
		score := 0.0
		if s, ok := c["similarity"].(float64); ok {
			score = s
		}
		hits = append(hits, chunkHit{id: id, score: score})
	}
	return hits
}

// drillKeptNodes mirrors navigation.py _drill_kept_nodes: vector-beam BFS descent
// from the TOC roots, keeping top-K children per level by cosine (relative beam
// threshold) or by keyword relevance (absolute minimum), returning the kept
// nodes, the single-parent map, the kept-name set and the overall best score.
func drillKeptNodes(qvec []float64, terms []string, nodes []structureNode, rels []structureRel) ([]structureNode, map[string]string, map[string]bool, float64) {
	byName, children, parents, roots := buildTocTree(nodes, rels)
	if len(roots) == 0 {
		return nil, nil, nil, 0.0
	}
	frontier := roots
	keptNames := map[string]bool{}
	depth := 0
	bestOverall := 0.0
	for len(frontier) > 0 && depth <= structMaxDepth {
		type scored struct {
			score float64
			name  string
		}
		var scoredList []scored
		for _, n := range frontier {
			e, ok := byName[n]
			if ok {
				scoredList = append(scoredList, scored{nodeScore(qvec, terms, e), n})
			}
		}
		if len(scoredList) == 0 {
			break
		}
		sort.SliceStable(scoredList, func(i, j int) bool { return scoredList[i].score > scoredList[j].score })
		best := scoredList[0].score
		if best > bestOverall {
			bestOverall = best
		}
		if len(qvec) == 0 && best < float64(structRelevanceMin) {
			break
		}
		var top []scored
		if len(qvec) > 0 {
			for _, s := range scoredList {
				if s.score >= best*structVecBeamRatio {
					top = append(top, s)
					if len(top) >= structBranchK {
						break
					}
				}
			}
		} else {
			for _, s := range scoredList {
				if s.score >= float64(structRelevanceMin) {
					top = append(top, s)
					if len(top) >= structBranchK {
						break
					}
				}
			}
		}
		if len(top) == 0 {
			break
		}
		var next []string
		for _, s := range top {
			if !keptNames[s.name] {
				keptNames[s.name] = true
			}
			next = append(next, children[s.name]...)
		}
		frontier = next
		depth++
	}
	for name := range keptNames {
		cur := parents[name]
		guard := 0
		for cur != "" && !keptNames[cur] && guard < structMaxDepth {
			keptNames[cur] = true
			cur = parents[cur]
			guard++
		}
	}
	var kept []structureNode
	for n := range keptNames {
		if e, ok := byName[n]; ok {
			kept = append(kept, e)
		}
	}
	return kept, parents, keptNames, bestOverall
}

// outlineStats mirrors navigation.py _outline_stats: the entity / chunk-pointer
// counts for the flat fallback outline (no drill happened => top_score 0).
func outlineStats(nodes []structureNode) (nodeCount, ptrCount int) {
	count := len(nodes)
	if count > structMaxNodes {
		count = structMaxNodes
	}
	ptrs := 0
	for _, n := range nodes[:count] {
		ptrs += len(collectChunkIDs([]structureNode{n}, 32))
	}
	return count, ptrs
}

// navTypeOr returns a node type, defaulting to "other" as Python's
// “(e.get("type") or "other")“ does.
func navTypeOr(t string) string {
	if t == "" {
		return "other"
	}
	return t
}

// navOutlineLine renders one "- name (type): desc [chunks: c1,c2]" line.
func navOutlineLine(indent, name, nodeType, desc string, chunks []string) string {
	line := indent + "- " + name + " (" + navTypeOr(nodeType) + ")"
	if desc != "" {
		line += ": " + Snippet(desc, structDescSnippet)
	}
	if c := chunkPtrs(chunks); c != "" {
		line += " [chunks: " + c + "]"
	}
	return line
}

// renderOutline mirrors navigation.py _render_outline: a compact flat outline of
// entities then relations (fallback when no query terms are usable).
func renderOutline(nodes []structureNode, rels []structureRel) string {
	var lines []string
	capE, capR := 40, 40
	for i, n := range nodes {
		if i >= capE || n.name == "" {
			continue
		}
		lines = append(lines, navOutlineLine("", n.name, n.nodeType, n.desc, n.sourceChunkIDs))
	}
	for i, r := range rels {
		if i >= capR || r.from == "" || r.to == "" {
			continue
		}
		rt := r.relType
		if rt == "" {
			rt = "related_to"
		}
		lines = append(lines, "- "+r.from+" -["+rt+"]-> "+r.to)
	}
	return strings.Join(lines, "\n")
}

// rankChunksByTerms mirrors navigation.py _rank_chunks_by_terms: rank candidate
// chunks by how many query terms overlap with their text. Zero-LLM keyword
// relevance for the precise chunk_ids path. The accessor abstracts reading a
// chunk's text so the harness stays store-agnostic and testable.
func rankChunksByTerms(candidates []chunkWithText, queries []string) []chunkWithText {
	terms := make([]string, 0, 8)
	for _, q := range queries {
		for _, tok := range queryTerms(q) {
			if !sliceContains(terms, tok) {
				terms = append(terms, tok)
			}
		}
	}
	if len(terms) == 0 {
		return candidates
	}
	type scored struct {
		hits int
		c    chunkWithText
	}
	var scoredList []scored
	for _, c := range candidates {
		text := strings.ToLower(c.text)
		hits := 0
		for _, t := range terms {
			if strings.Contains(text, t) {
				hits++
			}
		}
		if hits > 0 {
			scoredList = append(scoredList, scored{hits, c})
		}
	}
	sort.SliceStable(scoredList, func(i, j int) bool { return scoredList[i].hits > scoredList[j].hits })
	out := make([]chunkWithText, 0, len(scoredList))
	for _, s := range scoredList {
		out = append(out, s.c)
	}
	return out
}

// chunkWithText pairs a chunk id with its text for structure snippet ranking.
type chunkWithText struct {
	id   string
	text string
}

// structureDrillout is the per-document result of the TOC drill-down, mirroring
// what Python's _render_toc_drilldown returns and the per-doc stats
// _navigate_structure_impl aggregates. selector records which strategy chose the
// nodes: llm_toc / chunk_retrieval / beam (stats["selector"] in Python).
type structureDrillout struct {
	outline    string
	nodes      int
	chunkPtrs  int
	topScore   float64
	chunkPaths map[string]string
	selector   string
}

// chunkHit is a retrieved chunk id with its retrieval score, the input shape of
// the chunk_retrieval strategy (Python chunk_hits: list[(chunk_id, score)]).
type chunkHit struct {
	id    string
	score float64
}

// drillWithAncestors pulls the ancestors of names into the kept set so a path
// renders as root -> ... -> node, mirroring _with_ancestors in _render_toc_drilldown.
func drillWithAncestors(names map[string]bool, parents map[string]string) map[string]bool {
	for name := range names {
		cur := parents[name]
		guard := 0
		for cur != "" && !names[cur] && guard < structTocMaxDepth {
			names[cur] = true
			cur = parents[cur]
			guard++
		}
	}
	return names
}

// renderTocDrilldown mirrors navigation.py _render_toc_drilldown: it renders a
// query-focused outline of one document's compiled structure, choosing the nodes
// to show by one of three strategies in priority order:
//
//   - selected (non-empty): node names picked by the whole-TOC LLM pass, kept
//     whole with their ancestors — selector "llm_toc".
//   - chunkHits (non-empty): [(id, score)] from chunk retrieval (RAPTOR blob path);
//     the chunks are shown as retrieved and the tree only labels where they sit —
//     selector "chunk_retrieval".
//   - neither: VECTOR BEAM descent toward the query, falling back to keyword
//     overlap when no vectors are available — selector "beam".
//
// When no terms/vector and neither selection is provided it falls back to the flat
// outline. loader, when non-nil, fetches chunk texts so short snippets can be
// appended; a nil loader skips the snippet pass.
func renderTocDrilldown(query string, qvec []float64, nodes []structureNode, rels []structureRel, loader func(ids []string) []chunkWithText, chunkHits []chunkHit, selected []string) structureDrillout {
	flat := func() structureDrillout {
		out := renderOutline(nodes, rels)
		n, p := outlineStats(nodes)
		return structureDrillout{outline: out, nodes: n, chunkPtrs: p, topScore: 0.0, chunkPaths: map[string]string{}}
	}
	terms := queryTerms(query)
	// Pinned nodes (chunk retrieval / whole-TOC) stand even when the query yields
	// no usable terms or vector; bailing out here would discard a valid selection.
	if len(chunkHits) == 0 && len(selected) == 0 && len(terms) == 0 && len(qvec) == 0 {
		return flat()
	}

	byName, _, parents, _ := buildTocTree(nodes, rels)

	var kept []structureNode
	var keptNames map[string]bool
	var selectedIDs []string
	best := 0.0
	selector := "beam"

	switch {
	case len(selected) > 0:
		names := make(map[string]bool)
		for _, n := range selected {
			if _, ok := byName[n]; ok {
				names[n] = true
			}
		}
		keptNames = drillWithAncestors(names, parents)
		for n := range keptNames {
			if e, ok := byName[n]; ok {
				kept = append(kept, e)
			}
		}
		best = 0.0
		selector = "llm_toc"
	case len(chunkHits) > 0:
		seen := make(map[string]bool)
		for _, h := range chunkHits {
			if h.id != "" && !seen[h.id] {
				seen[h.id] = true
				selectedIDs = append(selectedIDs, h.id)
			}
		}
		covering := nodesCoveringChunks(nodes, selectedIDs)
		names := make(map[string]bool)
		for _, n := range covering {
			if _, ok := byName[n]; ok {
				names[n] = true
			}
		}
		keptNames = drillWithAncestors(names, parents)
		for n := range keptNames {
			if e, ok := byName[n]; ok {
				kept = append(kept, e)
			}
		}
		for _, h := range chunkHits {
			if h.score > best {
				best = h.score
			}
		}
		selector = "chunk_retrieval"
	default:
		kept, parents, keptNames, best = drillKeptNodes(qvec, terms, nodes, rels)
		selector = "beam"
	}
	if len(kept) == 0 && len(selectedIDs) == 0 {
		// A strategy was chosen but kept nothing; still report which one ran.
		out := flat()
		out.selector = selector
		return out
	}
	// depth_of: node -> number of kept ancestors (rendering indentation).
	depthOf := map[string]int{}
	for n := range keptNames {
		d := 0
		cur := parents[n]
		for cur != "" && keptNames[cur] {
			d++
			cur = parents[cur]
		}
		depthOf[n] = d
	}
	// root -> ... -> node path for each kept node, mapped onto its chunks.
	chunkPaths := map[string]string{}
	for _, e := range kept {
		name := strings.TrimSpace(e.name)
		if name == "" {
			continue
		}
		var segs []string
		segs = append(segs, name)
		cur := parents[name]
		guard := 0
		for cur != "" && guard < structMaxDepth {
			segs = append(segs, cur)
			cur = parents[cur]
			guard++
		}
		for i, j := 0, len(segs)-1; i < j; i, j = i+1, j-1 {
			segs[i], segs[j] = segs[j], segs[i]
		}
		path := strings.Join(segs, " -> ")
		for _, cid := range e.sourceChunkIDs {
			if cid != "" {
				if _, seen := chunkPaths[cid]; !seen {
					chunkPaths[cid] = path
				}
			}
		}
	}
	var lines []string
	count := len(kept)
	if count > structMaxNodes {
		count = structMaxNodes
	}
	for _, e := range kept[:count] {
		name := strings.TrimSpace(e.name)
		if name == "" {
			continue
		}
		indent := strings.Repeat("  ", depthOf[name])
		lines = append(lines, navOutlineLine(indent, name, e.nodeType, e.desc, e.sourceChunkIDs))
	}
	// Chunk snippets. When chunk retrieval did the selecting, the hits are shown
	// as-is (filtering them back through the nodes would drop chunks no node
	// covers); otherwise take the chunks behind the drilled nodes.
	wanted := collectChunkIDs(kept, 32)
	if len(selectedIDs) > 0 {
		wanted = selectedIDs
	}
	if len(wanted) > 0 && loader != nil {
		chunks := loader(wanted)
		if len(chunks) > 0 {
			var ranked []chunkWithText
			limit := structMaxChunks
			if len(selectedIDs) > 0 {
				// Retrieval order is the relevance order; re-ranking by terms would
				// discard the score that selected them.
				pos := make(map[string]int, len(selectedIDs))
				for i, id := range selectedIDs {
					pos[id] = i
				}
				ranked = append(ranked, chunks...)
				sort.SliceStable(ranked, func(i, j int) bool { return pos[ranked[i].id] < pos[ranked[j].id] })
				limit = structMaxChunkHits
			} else {
				ranked = rankChunksByTerms(chunks, []string{query})
			}
			capN := len(ranked)
			if capN > limit {
				capN = limit
			}
			for _, c := range ranked[:capN] {
				text := strings.TrimSpace(c.text)
				if text == "" {
					continue
				}
				lines = append(lines, "- [chunk "+c.id+"]: "+Snippet(text, 300))
			}
		}
	}
	return structureDrillout{
		outline:    strings.Join(lines, "\n"),
		nodes:      len(kept),
		chunkPtrs:  len(wanted),
		topScore:   best,
		chunkPaths: chunkPaths,
		selector:   selector,
	}
}

// ---------------------------------------------------------------------------
// navigate_structure (zero-LLM vector-beam drill-down; mirrors _navigate_structure_impl)
// ---------------------------------------------------------------------------

// structureKindsFor maps a navigate_structure kind string to the compiled-kinds
// set to read, mirroring navigation.py _structure_kinds_for. Defaults to catalog.
func structureKindsFor(kind string) map[string]bool {
	k := strings.TrimSpace(strings.ToLower(kind))
	if k == "" {
		k = "catalog"
	}
	switch k {
	case "mindmap", "mind_map", "concept":
		out := make(map[string]bool, len(mindmapKinds))
		for kk := range mindmapKinds {
			out[kk] = true
		}
		return out
	case "graph", "kg", "entity", "ontology":
		return map[string]bool{"graph": true, "ontology": true, "entity": true, "raptor": true}
	default:
		out := make(map[string]bool, len(catalogKinds))
		for kk := range catalogKinds {
			out[kk] = true
		}
		return out
	}
}

// structureNodesFromEntities converts loaded structure entities into the
// drill-down node shape (no vectors unless filled in by a vector pass).
func structureNodesFromEntities(es []structureEntity) []structureNode {
	if len(es) == 0 {
		return nil
	}
	nodes := make([]structureNode, 0, len(es))
	for _, e := range es {
		if strings.TrimSpace(e.Name) == "" {
			continue
		}
		nodes = append(nodes, structureNode{
			name:           strings.TrimSpace(e.Name),
			nodeType:       e.Type,
			desc:           e.Description,
			sourceChunkIDs: append([]string(nil), e.SourceChunkIDs...),
			vec:            e.Vec,
		})
	}
	return nodes
}

// structureNodeLoader fetches chunk texts by id for the drill-down snippet pass,
// mirroring the chunk loader _render_toc_drilldown uses. Returns nil when the
// store/engine is not available (outline still renders, minus snippets).
func structureNodeLoader(ctx context.Context, tenantID string, ids []string) []chunkWithText {
	raw := loadChunksByIDs(ctx, tenantID, ids)
	if len(raw) == 0 {
		return nil
	}
	out := make([]chunkWithText, 0, len(raw))
	for _, c := range raw {
		id := ChunkIDOf(c)
		if id == "" {
			continue
		}
		out = append(out, chunkWithText{id: id, text: ChunkTextOf(c)})
	}
	return out
}

// navigateStructureDoc mirrors Python _navigate_structure_impl for a single
// document: read its compiled structure of the requested kind, run the zero-LLM
// vector-beam drill-down, and assemble the <structure_navigation> XML. No chat
// LLM is involved. A nil query vector (embedder unavailable) drills by keyword
// relevance, exactly like Python's no-vector path.
//
// The returned NavResult carries the XML in Text and the reached document in
// DocIDs; the structureDrillout holds the stats (nodes kept, chunk pointers,
// top score) that the executor lifts into ToolOutcome.Metrics. EmptyReason is
// "no_structure" when the document has no compiled structure of the kind,
// "bad_args" when the query/doc is missing.
func navigateStructureDoc(ctx context.Context, tenantID, query, docID, kind string, deps SearchDeps) (NavResult, structureDrillout) {
	fail := func(reason, label string) (NavResult, structureDrillout) {
		return NavResult{
			Text:        `<structure_navigation count="0" error="` + label + `">` + "\n</structure_navigation>",
			EmptyReason: reason,
		}, structureDrillout{chunkPaths: map[string]string{}}
	}
	if query == "" {
		return fail(ReasonBadArgs, "query is required")
	}
	if docID == "" {
		return fail(ReasonNoDoc, "no document located")
	}
	// Python _load_compiled_structure → _resolve_doc_tenant: a document that is
	// not in the bound datasets yields an empty structure rather than an
	// unscoped read, so a stale doc_id cannot pull in another dataset's outline.
	if belongs, verified := docInDatasets(ctx, deps, docID); verified && !belongs {
		return fail(ReasonNoStructure, "no structure")
	}
	kinds := structureKindsFor(kind)
	// Vector-aware load mirrors _read_structures: encode the query first so the
	// structure rows can be read with the q_<dim>_vec field, then judge whether
	// the nodes carry per-node embeddings.
	var qvec []float64
	var vecField string
	if vec := encodeSeedVector(ctx, deps, tenantID, query); vec != nil {
		qvec = vec
		vecField = "q_" + strconv.Itoa(len(vec)) + "_vec"
	}
	nodes := structureNodesFromEntities(loadStructureEntities(ctx, tenantID, docID, kinds, vecField))
	if len(nodes) == 0 {
		return fail(ReasonNoStructure, "no structure")
	}
	loader := func(ids []string) []chunkWithText { return structureNodeLoader(ctx, tenantID, ids) }

	// RAPTOR / shared-vector blobs cannot score their nodes against the query, so
	// they defer to chunk retrieval for the drill (mirrors Python routing them to
	// _recall_chunk_ids_in_doc). Per-node-vector structures keep the beam drill.
	var chunkHits []chunkHit
	if !hasDistinctNodeVectors(nodes) {
		chunkHits = recallChunkIDsInDoc(ctx, deps.Backend, query, docID, tenantID, deps.KbIDs, 0)
	}
	drill := renderTocDrilldown(query, qvec, nodes, nil, loader, chunkHits, nil)

	esc := func(s string) string { return XMLEscape(s) }
	var parts []string
	parts = append(parts, fmt.Sprintf(`<structure_navigation count="1" query="%s" kind="%s">`, esc(query), esc(kind)))
	parts = append(parts, fmt.Sprintf(`  <doc rank="1" doc_id="%s" entities="%d" relations="0">`, esc(docID), len(nodes)))
	if drill.outline != "" {
		parts = append(parts, "    <structure>"+esc(drill.outline)+"</structure>")
	}
	parts = append(parts, "  </doc>", "</structure_navigation>")

	res := NavResult{
		Text:        strings.Join(parts, "\n"),
		DocIDs:      []string{docID},
		Entities:    entityMaps(nodes),
		ChunkPaths:  drill.chunkPaths,
		EmptyReason: "",
	}
	return res, drill
}

// entityMaps renders drill nodes back to the generic maps the orchestrator's
// NavResult.Entities slice carries.
func entityMaps(nodes []structureNode) []map[string]any {
	out := make([]map[string]any, 0, len(nodes))
	for _, n := range nodes {
		out = append(out, map[string]any{
			"name":             n.name,
			"type":             n.nodeType,
			"description":      n.desc,
			"source_chunk_ids": n.sourceChunkIDs,
		})
	}
	return out
}
