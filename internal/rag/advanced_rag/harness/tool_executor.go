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

// Package harness holds the low-level agentic-RAG capability library: the
// retrieval backend, the action-session runtime, the tool executor, and the
// compiled-structure / knowledge-graph navigation. These are the leaf building
// blocks the RAGTools methods in the parent `agent` package (Run, ComposeAnswer,
// ComposeNaiveAnswer, Formalize, ...) orchestrate. The split mirrors Python's
// layout, where agentic_rag.py's RAGTools imports the helpers under
// harness/ rather than inlining them.
//
// tool_executor.go is the Go-side tool-executor adapter: its
// searchExecutor type and the Execute dispatch correspond to the _exec_* tool
// methods in Python's harness/action_session.py (e.g. _exec_retrieve,
// _exec_navigate_tree, _exec_navigate_structure, _exec_list_chunks,
// _exec_calculate, _exec_web_search, _exec_wiki_query, _exec_graph_explore).
// In Python those live inline inside action_session.py; in Go they are pulled
// out into this file so the harness package need not import the advanced_rag package.
package harness

import (
	"context"
	"fmt"
	"strings"

	"ragflow/internal/agent/runtime"
	"ragflow/internal/service/nav"
)

// RunRequest is one harness run. It lives in the harness package (not the agent
// package) because searchExecutor needs it to carry the per-run query, and the
// advanced_rag package must not be imported from here (import-cycle rule). RAGTools.Run
// in the advanced_rag package uses it via the harness package.
type RunRequest struct {
	// Question is the user's question.
	Question string
	// ThinkingMode selects the ModeSpec ("low"/"medium"/"high"/"ultra").
	// An unrecognised value degrades to NAIVE, as in Python.
	ThinkingMode string
	// Keywords narrow retrieved chunks to the sentences mentioning them.
	Keywords string
	// DatasetIDs restricts retrieval. Empty falls back to the canvas context.
	DatasetIDs []string
	// TenantID scopes retrieval. Empty falls back to the canvas context.
	TenantID string
	// UseCompiled enables compiled-structure expansion during retrieval.
	UseCompiled bool
	// TopN is the per-search result count. <=0 selects the default.
	TopN int
	// DeadlineLeft is the wall-clock budget in seconds. <=0 selects the default.
	DeadlineLeft float64
	// MaxLength is the chat model's context window (Python
	// tools.chat_mdl.max_length). It bounds evidence and document-level reads.
	// <=0 selects the file-level defaults.
	MaxLength int
	// SessionID identifies the conversation this call belongs to. It is the
	// scope key for cross-turn state such as the near-duplicate answer cache:
	// calls sharing a session reuse one cache, calls without it get their own.
	SessionID string
}

// searchExecutor adapts SearchDeps to ToolExecutor, so the retriever and
// compiled expander are what the session's retrieval tools call.
type searchExecutor struct {
	deps SearchDeps
	req  RunRequest
}

// NewSearchExecutor builds the retrieval/navigation ToolExecutor used by both
// the single-session path and the agentic loop's programmatic fan-out fetches.
//
// Exported so the advanced_rag package (which cannot be imported from here) can reuse
// the same evidence handling instead of duplicating it.
func NewSearchExecutor(deps SearchDeps, req RunRequest) ToolExecutor {
	return &searchExecutor{deps: deps, req: req}
}

// Execute implements ToolExecutor for the wired tools. Tools whose port has not
// landed are classified, not errored: they report MISS (the tool is valid, this
// call reached nothing) so the model falls back to a different tool instead of
// stalling.
func (e *searchExecutor) Execute(ctx context.Context, name string, args map[string]any) (ToolOutcome, error) {
	switch name {
	case "retrieve", "search_chunks", "grep_search", "grep_chunks":
		return e.search(ctx, name, args)
	case "navigate_tree":
		return e.navigateTree(ctx, args)
	case "navigate_structure":
		return e.navigateStructure(ctx, args)
	case "list_chunks":
		return e.listChunks(ctx, args)
	case "fetch_full_document":
		return e.fetchFullDocument(ctx, args)
	case "summarize_document":
		return e.summarizeDocument(ctx, args)
	case "calculate":
		return e.calculate(ctx, args)
	case "graph_explore":
		return e.graphExplore(ctx, args)
	case "web_search":
		return e.webSearch(ctx, args)
	case "wiki_query":
		return e.wikiQuery(ctx, args)
	}
	return ToolOutcome{
		Payload: []any{map[string]any{
			"kind": name,
			"note": fmt.Sprintf("%s is not wired in this deployment yet. Use retrieve, search_chunks, navigate_tree or navigate_structure.", name),
		}},
		Status:  StatusMiss,
		Reason:  ReasonNoDoc,
		Metrics: map[string]any{},
	}, nil
}

// navigateTree routes to the top-n documents by descending the dataset's
// compiled navigation tree.
//
// It reports three distinct outcomes so the navigation ladder can act on WHAT
// happened:
//   - no router configured → EMPTY/no_structure (dataset-level: disable the tool)
//   - tree exists but this query routed to nothing → MISS (query-level: fall
//     through to `global`, do NOT disable)
//   - routed → OK with the doc ids and their summaries.
func (e *searchExecutor) fetchFullDocument(ctx context.Context, args map[string]any) (ToolOutcome, error) {
	docID := argString(args, "doc_id")
	if docID == "" {
		return ToolOutcome{Payload: []any{}, Status: StatusError, Reason: ReasonBadArgs}, nil
	}
	if e.deps.DocChunks == nil {
		return ToolOutcome{
			Payload: []any{map[string]any{
				"kind":   "fetch_full_document",
				"doc_id": docID,
				"note":   "Whole-document reading is not available for this session.",
			}},
			Status: StatusEmpty,
			Reason: ReasonNoDoc,
		}, nil
	}
	chunks, aggs := fetchFullDocument(ctx, e.deps, docID, e.req.MaxLength)
	if len(chunks) == 0 {
		return ToolOutcome{Payload: []any{}, Status: StatusEmpty, Reason: ReasonNoDoc}, nil
	}
	return ToolOutcome{
		Payload: []any{map[string]any{
			"kind":      "fetch_full_document",
			"doc_id":    docID,
			"count":     len(chunks),
			"chunks":    chunks,
			"doc_aggs":  aggs,
			"doc_names": []string{docID},
		}},
		Status:  StatusOK,
		Metrics: map[string]any{"chunks": len(chunks)},
	}, nil
}

// summarizeDocument mirrors Python's summarize_document tool: load the document
// and hand the model the freshly rendered evidence blocks.
func (e *searchExecutor) summarizeDocument(ctx context.Context, args map[string]any) (ToolOutcome, error) {
	docID := argString(args, "doc_id")
	if docID == "" {
		return ToolOutcome{Payload: []any{}, Status: StatusError, Reason: ReasonBadArgs}, nil
	}
	if e.deps.DocChunks == nil {
		return ToolOutcome{
			Payload: []any{map[string]any{
				"kind":   "summarize_document",
				"doc_id": docID,
				"note":   "Whole-document reading is not available for this session.",
			}},
			Status: StatusEmpty,
			Reason: ReasonNoDoc,
		}, nil
	}
	blocks := summarizeDocument(ctx, e.deps, docID, e.req.MaxLength)
	if len(blocks) == 0 {
		return ToolOutcome{Payload: []any{}, Status: StatusEmpty, Reason: ReasonNoDoc}, nil
	}
	content := ""
	for _, b := range blocks {
		content += b + "\n\n"
	}
	return ToolOutcome{
		Payload: []any{map[string]any{
			"kind":    "summarize_document",
			"doc_id":  docID,
			"content": content,
		}},
		Status:  StatusOK,
		Metrics: map[string]any{"blocks": len(blocks)},
	}, nil
}

func (e *searchExecutor) navigateTree(ctx context.Context, args map[string]any) (ToolOutcome, error) {
	query := argString(args, "query")
	if query == "" {
		return ToolOutcome{Payload: []any{}, Status: StatusError, Reason: ReasonBadArgs}, nil
	}
	router := e.deps.NavRouter
	if router == nil {
		// Default navigation-tree router mirrors Python _NAV_TREE_ROUTER = "chunk_agg":
		// route documents by aggregating raw-chunk retrieval rather than by a nav-row
		// KNN descent (nav.NewNavServiceRouter). A nil retrieval backend degrades to the
		// nav-row router, which needs no backend.
		if e.deps.Backend != nil {
			router = nav.NewChunkAggRouter(
				chunkAggRetrieveFrom(e.deps.Backend),
				nav.ChunkAggSummarize(),
			)
		} else {
			router = nav.NewNavServiceRouter()
		}
	}
	res := NavigateTree(ctx, router, NavTreeInput{
		Query:    query,
		Keywords: e.req.Keywords,
		DocScope: toolDocScope(args),
		TenantID: e.deps.TenantID,
		KbIDs:    e.deps.KbIDs,
	})

	switch res.EmptyReason {
	case ReasonNoStructure:
		return ToolOutcome{
			Payload: []any{map[string]any{
				"kind":    "navigate_tree",
				"note":    "This dataset has no compiled document-navigation tree. Use search_chunks / retrieve instead.",
				"query":   query,
				"doc_ids": []string{},
			}},
			Status:  StatusEmpty,
			Reason:  ReasonNoStructure,
			Metrics: map[string]any{"docs": 0, "routed_docs": [][2]string{}},
		}, nil
	case ReasonNoDoc:
		// Structure exists, this query reached nothing — a MISS, not an EMPTY.
		return ToolOutcome{
			Payload: []any{map[string]any{
				"kind":    "navigate_tree",
				"content": res.Text,
				"doc_ids": []string{},
				"note":    "The navigation tree exists but this query routed to no document. Try a different topic/entity phrasing, or use search_chunks.",
				"query":   query,
			}},
			Status:  StatusMiss,
			Reason:  ReasonNoDoc,
			Metrics: map[string]any{"docs": 0},
		}, nil
	case ReasonInfra, ReasonBadArgs:
		return ToolOutcome{
			Payload: []any{},
			Status:  ReasonStatus(res.EmptyReason),
			Reason:  res.EmptyReason,
		}, nil
	}

	// Routed: expose the summary-bearing payload the ladder consumes.
	payload := []any{map[string]any{
		"kind":    "navigate_tree",
		"content": res.Text,
		"doc_ids": res.DocIDs,
	}}
	return ToolOutcome{
		Payload:     payload,
		EvidenceIDs: nil, // routing only — no passages retrieved
		Status:      StatusOK,
		Reason:      ReasonNone,
		Metrics: map[string]any{
			"docs":        len(res.DocIDs),
			"routed_docs": res.RoutedDocs,
		},
	}, nil
}

// chunkAggRetrieveFrom adapts a harness Retriever into the chunk_agg retrieve
// leg. It scopes the search to the dataset (kbID) and the caller's document
// scope and pulls the wide pool chunk_agg needs; the backend is expected to
// exclude compiled rows (the production retrieval filters available_int=1),
// mirroring Python settings.retriever.retrieval under _search_layers_nav_chunk_agg.
//
// It stays in the harness package because it depends on the agentic Retriever /
// RetrieveRequest; the routing algorithm itself lives in internal/service/nav.
func chunkAggRetrieveFrom(r Retriever) nav.ChunkRetriever {
	return func(ctx context.Context, tenantID, kbID, query string, docScope []string, topN int, _ float64) []map[string]any {
		chunks, err := r.Retrieve(ctx, RetrieveRequest{
			Query:      query,
			DatasetIDs: []string{kbID},
			TenantID:   tenantID,
			DocScope:   docScope,
			TopN:       topN,
		})
		if err != nil || len(chunks) == 0 {
			return nil
		}
		return chunks
	}
}

// navigateStructure pinpoints passages inside ONE document using its compiled
// structure (heading tree / concept mindmap).
//
// It delegates to the navigation package's NavigateStructure, which loads the
// document's compiled rows, asks the model which entities answer the query, and
// returns the underlying source chunks.
//
// NOTE on shapes: the Go reader (navigation.loadStructureEntities) currently
// reads only the compact graph-blob rows, whereas Python merges BOTH row shapes
// (graph blob AND per-entity/relation rows). ParseCompiledStructure in
// navtools.go already implements the merged parsing; wiring it into the reader
// is the remaining step for full parity.
func (e *searchExecutor) navigateStructure(ctx context.Context, args map[string]any) (ToolOutcome, error) {
	query := argString(args, "query")
	docID := argString(args, "doc_id")
	if docID == "" {
		// The schema requires doc_id; a scope list is also accepted by the model.
		if scope := toolDocScope(args); len(scope) > 0 {
			docID = scope[0]
		}
	}
	// Only doc_id is required (mirrors Python's schema: {"required":
	// ["doc_id"]}). A missing query is allowed — the structure read then simply
	// selects nothing, which surfaces as EMPTY rather than an argument error.
	if docID == "" {
		return ToolOutcome{Payload: []any{}, Status: StatusError, Reason: ReasonBadArgs}, nil
	}
	if query == "" {
		query = e.req.Question
	}

	kind := argString(args, "kind")
	if kind == "" {
		kind = "catalog"
	}
	// Zero-LLM vector-beam drill-down (mirrors Python _exec_navigate_structure +
	// _navigate_structure_impl): read the document's compiled structure of the
	// requested kind, drill toward the query, and hand the model the
	// <structure_navigation> outline with chunk-pointer anchors.
	res, drill := navigateStructureDoc(ctx, e.deps.TenantID, query, docID, kind, e.deps)
	metrics := map[string]any{
		"entities":    drill.nodes,
		"chunk_ptrs":  drill.chunkPtrs,
		"top_score":   drill.topScore,
		"chunk_paths": drill.chunkPaths,
	}
	if res.EmptyReason != "" {
		return ToolOutcome{
			Payload: []any{map[string]any{
				"kind":   "navigate_structure",
				"doc_id": docID,
				"note":   fmt.Sprintf("No compiled structure of kind=%q reachable for this document. Try another doc_id or kind, or use search_chunks / retrieve / list_chunks.", kind),
			}},
			Status:  ReasonStatus(res.EmptyReason),
			Reason:  res.EmptyReason,
			Metrics: metrics,
		}, nil
	}
	// Reached a structure but drilled to nothing usable: not an error, just weak —
	// the orchestrator falls back to retrieval on status == poor.
	status := StatusOK
	if drill.chunkPtrs == 0 {
		status = StatusPoor
	}
	content := res.Text
	if len(content) > 8000 {
		content = content[:8000]
	}
	evidenceIDs := res.DocIDs
	if len(evidenceIDs) == 0 && docID != "" {
		evidenceIDs = []string{docID}
	}
	return ToolOutcome{
		Payload:     []any{map[string]any{"kind": "navigate_structure", "doc_id": docID, "content": content}},
		EvidenceIDs: evidenceIDs,
		Status:      status,
		Reason:      ReasonNone,
		Metrics:     metrics,
	}, nil
}

// argString reads a string arg, treating an absent key or an explicit JSON null
// as empty. fmt.Sprint(nil) yields "<nil>", so the presence check must come
// first — otherwise a missing arg silently becomes the literal "<nil>".
func argString(args map[string]any, key string) string {
	raw, ok := args[key]
	if !ok || raw == nil {
		return ""
	}
	s := strings.TrimSpace(fmt.Sprint(raw))
	if s == "<nil>" {
		return ""
	}
	return s
}

// listChunks deep-reads the FULL text of one document from the accumulated
// evidence pool (Python lists the document's chunks in reading order).
//
// Only chunks already retrieved are available: the Go port has no direct
// document-store read, so an unknown doc_id is a MISS (query-level) rather than
// an EMPTY — the tool is fine, this particular id simply is not in evidence.
func (e *searchExecutor) listChunks(_ context.Context, args map[string]any) (ToolOutcome, error) {
	// NOTE: fmt.Sprint(nil) yields "<nil>", not "", so an absent arg would pass
	// a naive empty check. Resolve presence first, then stringify.
	rawID, hasID := args["doc_id"]
	docID := ""
	if hasID && rawID != nil {
		docID = strings.TrimSpace(fmt.Sprint(rawID))
	}
	if docID == "" || docID == "<nil>" || e.deps.KB == nil {
		return ToolOutcome{Payload: []any{}, Status: StatusError, Reason: ReasonBadArgs}, nil
	}
	var payload []any
	var evidenceIDs []string
	for i, c := range e.deps.KB.Chunks {
		if DocIDOf(c) != docID {
			continue
		}
		// Cap at 30 chunks per Python action_session._exec_list_chunks ([:30]):
		// the model reads them in order and a single doc can dwarf the context.
		if len(payload) >= 30 {
			break
		}
		payload = append(payload, map[string]any{
			"id":      ChunkIDOf(c),
			"doc_id":  docID,
			"content": ChunkTextOf(c),
		})
		evidenceIDs = append(evidenceIDs, fmt.Sprint(i))
	}
	if len(payload) == 0 {
		return ToolOutcome{
			Payload:     []any{},
			EvidenceIDs: nil,
			Status:      StatusMiss,
			Reason:      ReasonNoDoc,
			Metrics:     map[string]any{"hits": 0},
		}, nil
	}
	return ToolOutcome{
		Payload:     payload,
		EvidenceIDs: evidenceIDs,
		Status:      StatusOK,
		Metrics:     map[string]any{"hits": len(payload)},
	}, nil
}

// search runs one retrieval call for a tool invocation.
//
// Python's retrieve/search_chunks both funnel into tools/search.py, differing
// only in whether compiled expansion runs (search_chunks expands, retrieve does
// not) and in the accepted query count.
func (e *searchExecutor) search(ctx context.Context, name string, args map[string]any) (ToolOutcome, error) {
	queries := toolQueries(args)
	if len(queries) == 0 {
		return ToolOutcome{Payload: []any{}, Status: StatusError, Reason: ReasonBadArgs}, nil
	}

	// Max queries per tool call mirrors Python action_session.execute_tool
	// (_arg_query_list): retrieve=3, search_chunks=2. grep_search/grep_chunks
	// are Go-internal tools (Python exposes no such session tool), so they are
	// left uncapped.
	maxQ := len(queries)
	switch name {
	case "retrieve":
		maxQ = 3
	case "search_chunks":
		maxQ = 2
	}
	if len(queries) > maxQ {
		queries = queries[:maxQ]
	}

	var payload []any
	var evidenceIDs []string
	newChunks := 0
	for _, q := range queries {
		// Per-tool top_n (mirrors Python action_session: retrieve=10,
		// search_chunks=20). They were previously collapsed onto e.req.TopN,
		// so search_chunks returned far fewer candidates than Python.
		topN := 10
		if name == "search_chunks" {
			topN = 20
		}
		chunks, aggs := HybridSearch(ctx, e.deps, SearchParams{
			Question:    q,
			Keywords:    e.req.Keywords,
			UseCompiled: name == "search_chunks" && e.req.UseCompiled,
			TopN:        topN,
			KbIDs:       e.req.DatasetIDs,
			DocScope:    toolDocScope(args),
			// search_chunks mirrors Python's standalone hybrid_search tool:
			// exclude compiled rows from the base retrieval and default the
			// vector weight to 0.3, unlike retrieve (RAGTools.retrieve).
			HybridTool: name == "search_chunks",
		})
		if len(chunks) == 0 {
			continue
		}
		// Mirror Python _run_search: only the first snippetsPerQuery (4) hits
		// of each query are admitted to the output AND the shared pool.
		// Python's table pass-through + 1200-char truncation happens inside
		// passageFromChunk, so the pool and payload stay aligned with Python.
		if len(chunks) > snippetsPerQuery {
			chunks = chunks[:snippetsPerQuery]
		}
		// Index the newly ADDED chunks by comparing the pool size across the
		// merge: Merge returns an index for every contributed chunk INCLUDING
		// duplicates, so len(idx) would report a hit even when nothing was new
		// (which is exactly the "searched again, learned nothing" case the
		// REDUNDANT status exists to catch).
		before := len(e.deps.KB.Chunks)
		idx := e.deps.KB.Merge(chunks, aggs)
		for _, i := range idx {
			evidenceIDs = append(evidenceIDs, fmt.Sprint(i))
		}
		newChunks += len(e.deps.KB.Chunks) - before
		for _, c := range chunks {
			payload = append(payload, passageFromChunk(c, q))
		}
	}

	if len(payload) == 0 {
		return ToolOutcome{
			Payload:     []any{},
			EvidenceIDs: nil,
			Status:      StatusMiss,
			Reason:      ReasonNoDoc,
			Metrics:     map[string]any{"queries": len(queries)},
		}, nil
	}
	// Zero new evidence (every hit was already in the pool) is REDUNDANT, not
	// OK: the model otherwise sees passages and concludes the search succeeded.
	if newChunks == 0 {
		return ToolOutcome{
			Payload:     []any{},
			EvidenceIDs: evidenceIDs,
			Status:      StatusRedundant,
			Reason:      ReasonNone,
			Metrics:     map[string]any{"hits": len(payload), "new_evidence": 0},
		}, nil
	}
	return ToolOutcome{
		Payload:     payload,
		EvidenceIDs: evidenceIDs,
		Status:      StatusOK,
		Reason:      ReasonNone,
		Metrics:     map[string]any{"hits": len(payload), "new_evidence": newChunks},
	}, nil
}

// calculate mirrors Python's calculate tool: derive a number the evidence does
// not state outright, by having the model write ONE expression and evaluating it
// against the AST whitelist (see arithmetic.go).
//
// The computed value is returned as evidence, so a later answer step can cite it
// without re-deriving. No derivation being needed (or possible) is a MISS, not an
// error — the model then answers from the facts it already has.
func (e *searchExecutor) calculate(ctx context.Context, args map[string]any) (ToolOutcome, error) {
	question := argString(args, "question")
	if question == "" {
		// The schema requires it, but a model may omit it; fall back to the run's
		// question so the derivation still targets what was asked.
		question = e.req.Question
	}
	facts := make([]string, 0, 8)
	for _, f := range toolStringList(args, "facts") {
		if s := strings.TrimSpace(f); s != "" {
			facts = append(facts, s)
		}
	}
	if question == "" || len(facts) == 0 {
		return ToolOutcome{Payload: []any{}, Status: StatusError, Reason: ReasonBadArgs}, nil
	}
	if e.deps.Model == nil {
		// Mirror Python action_session._exec_calculate: no chat model is an
		// infra failure, never a derivable-answer miss.
		return ToolOutcome{
			Payload: []any{map[string]any{
				"kind": "calculate",
				"note": "No model is available for numeric derivation. Answer from the facts you have.",
			}},
			Status:  StatusError,
			Reason:  ReasonInfra,
			Metrics: map[string]any{},
		}, nil
	}
	cf := ComputeFromFacts(ctx, e.deps.Model, question, facts, 0)
	if cf == nil {
		// Nothing derivable is a POOR (ran, but produced nothing usable), not a
		// query miss — mirrors Python's status=POOR/reason=no_doc for calculate.
		return ToolOutcome{
			Payload: []any{map[string]any{
				"kind":       "calculate",
				"expression": nil,
				"note":       "No derived number is needed (the facts already state it) or a figure is genuinely missing. Answer from the facts you have.",
			}},
			Status:  StatusPoor,
			Reason:  ReasonNoDoc,
			Metrics: map[string]any{},
		}, nil
	}
	fact := fmt.Sprintf("%s: %s (=%s)", cf.Label, cf.Value, cf.Expression)
	return ToolOutcome{
		Payload: []any{map[string]any{
			"kind":       "calculate",
			"content":    fact,
			"label":      cf.Label,
			"value":      cf.Value,
			"result":     cf.Value,
			"expression": cf.Expression,
			"uses":       cf.Uses,
		}},
		Status:  StatusOK,
		Reason:  ReasonNone,
		Metrics: map[string]any{"label": cf.Label, "value": cf.Value},
	}, nil
}

// webSearch implements the `web_search` tool. The web_search tool schema takes
// a 1-2 element array of query strings (see webSearchToolSpec), so args is the
// positional query list, not a key/value object.
func (e *searchExecutor) webSearch(ctx context.Context, args map[string]any) (ToolOutcome, error) {
	if e.deps.WebSearch == nil {
		// Mirror Python action_session._exec_web_search: an absent provider is
		// an infra ERROR with an explicit do-not-retry note (returning MISS
		// would let the model retry the same dead tool and burn turns), never a
		// query-level miss.
		return ToolOutcome{
			Payload: []any{map[string]any{
				"kind": "web_search",
				"note": "web_search is not configured in this deployment. Do not use this tool again; use the corpus tools (retrieve / search_chunks / navigate_*) instead.",
			}},
			Status:  StatusError,
			Reason:  ReasonInfra,
			Metrics: map[string]any{},
		}, nil
	}
	queries := toolStringList(args, "")
	if len(queries) == 0 {
		// Fall back to a single positional element pushed under the empty key.
		if raw, ok := args[""]; ok {
			queries = toolStringList(map[string]any{"q": raw}, "q")
		}
	}
	if len(queries) == 0 {
		return ToolOutcome{Payload: []any{}, Status: StatusError, Reason: ReasonBadArgs}, nil
	}
	// Cap at two queries (Python action_session.execute_tool passes
	// _arg_query_list(args, 2) for web_search).
	if len(queries) > 2 {
		queries = queries[:2]
	}
	results, err := e.deps.WebSearch.Search(ctx, queries)
	if err != nil || len(results) == 0 {
		return ToolOutcome{
			Payload: []any{map[string]any{
				"kind": "web_search",
				"note": "No web result returned.",
			}},
			Status:  StatusMiss,
			Reason:  ReasonNoDoc,
			Metrics: map[string]any{},
		}, nil
	}
	passages := make([]any, 0, len(results))
	for i, r := range results {
		passages = append(passages, map[string]any{
			"chunk_id": fmt.Sprintf("web_%d", i),
			"doc_id":   "web",
			"title":    "web",
			"content":  r,
			"query":    strings.Join(queries, " | "),
		})
	}
	return ToolOutcome{
		Payload: passages,
		Status:  StatusOK,
		Reason:  ReasonNone,
		Metrics: map[string]any{"n_web": len(passages)},
	}, nil
}

// toolStringList reads a string-list arg, tolerating []any, []string, and a
// single string (models emit all three).
func toolStringList(args map[string]any, key string) []string {
	raw, ok := args[key]
	if !ok || raw == nil {
		return nil
	}
	switch v := raw.(type) {
	case []string:
		return v
	case []any:
		out := make([]string, 0, len(v))
		for _, item := range v {
			if item == nil {
				continue
			}
			out = append(out, fmt.Sprint(item))
		}
		return out
	case string:
		return []string{v}
	}
	return nil
}
func toolQueries(args map[string]any) []string {
	raw, ok := args["query"]
	if !ok || raw == nil {
		// Absent (or an explicit null, which JSON models emit) falls back to the
		// `q` alias before giving up.
		if q, ok := args["q"].(string); ok && strings.TrimSpace(q) != "" {
			return []string{strings.TrimSpace(q)}
		}
		return nil
	}
	switch v := raw.(type) {
	case string:
		if strings.TrimSpace(v) == "" {
			return nil
		}
		return []string{v}
	case []any:
		out := make([]string, 0, len(v))
		for _, item := range v {
			if s := strings.TrimSpace(fmt.Sprint(item)); s != "" {
				out = append(out, s)
			}
		}
		return out
	case []string:
		out := make([]string, 0, len(v))
		for _, s := range v {
			if strings.TrimSpace(s) != "" {
				out = append(out, s)
			}
		}
		return out
	}
	return nil
}

// toolDocScope reads an optional doc_scope / doc_ids restriction.
func toolDocScope(args map[string]any) []string {
	raw, ok := args["doc_scope"]
	if !ok {
		raw = args["doc_ids"]
	}
	switch v := raw.(type) {
	case []any:
		out := make([]string, 0, len(v))
		for _, item := range v {
			if s := strings.TrimSpace(fmt.Sprint(item)); s != "" {
				out = append(out, s)
			}
		}
		return out
	case []string:
		return v
	case string:
		if s := strings.TrimSpace(v); s != "" {
			return []string{s}
		}
	}
	return nil
}

// passageFromChunk renders one chunk as the passage dict the model sees.
func passageFromChunk(c map[string]any, query string) map[string]any {
	// Table chunks pass through UN-truncated (mirrors Python _admit_evidence:
	// the 1200-char cap hides answer rows mid/late in a long standings table).
	// Only the model-facing text is truncated; the shared pool stores the full
	// chunk regardless.
	var content string
	if IsTableChunk(c) {
		content = ChunkTextOf(c)
	} else {
		content = Snippet(ChunkTextOf(c), 1200)
	}
	return map[string]any{
		"chunk_id": ChunkIDOf(c),
		"doc_id":   DocIDOf(c),
		"title":    DocTitleOf(c),
		"content":  content,
		"query":    query,
	}
}

// PublishReferences writes the accumulated evidence into the canvas state so the
// agent's post-stream citation grounding can read it.
//
// Chunk and aggregation shapes match runtime's referenceChunksFromRetrieval /
// referenceDocAggsFromRetrieval (which dual-write both Python and Go field
// names), so no translation is needed by the consumer. Exported so the agent
// package's Run can call it.
func PublishReferences(ctx context.Context, kb *Kbinfos) {
	state, _, err := runtime.GetStateFromContext[*runtime.CanvasState](ctx)
	if err != nil || state == nil {
		return
	}
	chunks := make([]map[string]any, 0, len(kb.Chunks))
	for idx, c := range kb.Chunks {
		chunkID := ChunkIDOf(c)
		docID := DocIDOf(c)
		name := DocTitleOf(c)
		content := ChunkTextOf(c)
		datasetID := DatasetIDOf(c)
		chunks = append(chunks, map[string]any{
			"id":                  fmt.Sprint(idx),
			"chunk_id":            chunkID,
			"content":             content,
			"content_with_weight": content,
			"document_id":         docID,
			"doc_id":              docID,
			"document_name":       name,
			"docnm_kwd":           name,
			"dataset_id":          datasetID,
			"kb_id":               datasetID,
		})
	}
	state.SetRetrievalReferences(chunks, kb.DocAggs)
}

// RuntimeRetriever adapts runtime.GetRetrievalService() to the harness Retriever
// interface. The service is read on every call (not captured at construction)
// because the server installs it during boot, which may happen after a harness
// component was built.
//
// This is the only place in the harness that knows about internal/agent/runtime;
// it lives in the harness root package (not a sub-package) so the runtime
// dependency does not leak into the harness's testable core.
type RuntimeRetriever struct{}

func (r *RuntimeRetriever) Retrieve(ctx context.Context, req RetrieveRequest) ([]map[string]any, error) {
	svc := runtime.GetRetrievalService()
	chunks, err := svc.Search(ctx, nil, runtime.RetrievalRequest{
		Query:                    req.Query,
		DatasetIDs:               req.DatasetIDs,
		DocScope:                 req.DocScope,
		TopN:                     req.TopN,
		TopK:                     req.TopK,
		RerankCandidatesCount:    req.RerankCandidatesCount,
		SimilarityThreshold:      &req.SimilarityThreshold,
		KeywordsSimilarityWeight: &req.KeywordsSimilarityWeight,
		TenantID:                 req.TenantID,
		RankFeature:              req.RankFeature,
		// ExcludeCompiled maps Python hybrid_search's
		// must_not={"exists":"compile_kwd"} onto the runtime request's
		// OnlyOriginalText (the "no compile_kwd" exclusion).
		OnlyOriginalText: req.ExcludeCompiled,
	})
	if err != nil {
		return nil, err
	}
	return chunksToMaps(chunks), nil
}

// chunksToMaps normalises runtime.RetrievalChunk values into the harness chunk
// shape (the keys the rest of the harness expects: chunk_id / content / doc_id
// / docnm_kwd / ...).
func chunksToMaps(chunks []runtime.RetrievalChunk) []map[string]any {
	out := make([]map[string]any, 0, len(chunks))
	for _, c := range chunks {
		out = append(out, map[string]any{
			"chunk_id":          c.ID,
			"content":           c.Content,
			"doc_id":            c.DocumentID,
			"docnm_kwd":         c.DocumentName,
			"dataset_id":        c.DatasetID,
			"image_id":          c.ImageID,
			"url":               c.URL,
			"positions":         c.Positions,
			"chunk_order_int":   c.ChunkIndex,
			"page_num_int":      c.PageNum,
			"score":             c.Score,
			"term_similarity":   c.TermSimilarity,
			"vector_similarity": c.VectorSimilarity,
		})
	}
	return out
}

// TenantIDFromContext returns the tenant id bound on the canvas context, used as
// a fallback for SearchDeps.TenantID when the caller's SearchRequest leaves it
// empty. The harness stays independent of the tool registry, so it reads only
// what CanvasState.GetVar exposes. Exported so the advanced_rag package's Run can call
// it.
func TenantIDFromContext(ctx context.Context) string {
	state, _, err := runtime.GetStateFromContext[*runtime.CanvasState](ctx)
	if err != nil || state == nil {
		return ""
	}
	if tid, err := state.GetVar("tenant_id"); err == nil {
		if s, _ := tid.(string); s != "" {
			return s
		}
	}
	return ""
}
