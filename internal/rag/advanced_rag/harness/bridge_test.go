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
	"strings"
	"testing"
)

// End-to-end wiring tests: one Run call, with a stubbed retriever and a
// scripted model, must move evidence from the backend into Kbinfos and (when a
// canvas state is attached) into the citation store.

// corpusRetriever answers every query from a fixed corpus.
type corpusRetriever struct{ calls []string }

func (c *corpusRetriever) Retrieve(_ context.Context, req RetrieveRequest) ([]map[string]any, error) {
	c.calls = append(c.calls, req.Query)
	return []map[string]any{{
		"chunk_id":   "c1",
		"content":    "Culdcept was created by OmiyaSoft and released in 1999.",
		"doc_id":     "doc-culdcept",
		"doc_name":   "Culdcept History",
		"dataset_id": "kb1",
	}}, nil
}

func TestSearchExecutorReportsMissAndRedundancy(t *testing.T) {
	kb := &Kbinfos{}
	sd := SearchDeps{Backend: &corpusRetriever{}, KbIDs: []string{"kb1"}, KB: kb}
	ex := &searchExecutor{deps: sd, req: RunRequest{}}

	// First call: OK with new evidence.
	oc, err := ex.Execute(context.Background(), "retrieve", map[string]any{"query": "q1"})
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if oc.Status != StatusOK {
		t.Fatalf("first call status = %s, want ok", oc.Status)
	}
	if len(oc.EvidenceIDs) != 1 {
		t.Errorf("evidence ids = %v, want 1", oc.EvidenceIDs)
	}

	// Same query again: every hit is already in the pool -> REDUNDANT, not ok.
	oc, _ = ex.Execute(context.Background(), "retrieve", map[string]any{"query": "q1"})
	if oc.Status != StatusRedundant {
		t.Errorf("repeat call status = %s, want redundant", oc.Status)
	}

	// Bad args -> error, so the model can correct itself.
	oc, _ = ex.Execute(context.Background(), "retrieve", map[string]any{})
	if oc.Status != StatusError || oc.Reason != ReasonBadArgs {
		t.Errorf("missing query: got (%s,%s), want (error,bad_args)", oc.Status, oc.Reason)
	}

	// A genuinely unwired tool -> miss (valid tool, nothing reached), never an
	// error, so the model falls back instead of stalling.
	oc, _ = ex.Execute(context.Background(), "wiki_query", map[string]any{"query": "topic"})
	if oc.Status != StatusMiss {
		t.Errorf("unwired tool status = %s, want miss", oc.Status)
	}
}

// fakeWikiRetriever returns a fixed compiled wiki page.
type fakeWikiRetriever struct{}

func (f *fakeWikiRetriever) SearchWiki(_ context.Context, question string, keywords []string, topN int) ([]WikiPage, error) {
	return []WikiPage{{
		ChunkID: "w1", DocID: "wdoc1", DocName: "Synthesis", Title: "Culdcept Overview",
		Content: "Culdcept is a board game by OmiyaSoft.", Score: 0.9,
	}}, nil
}

func TestWikiQueryWiredReturnsPages(t *testing.T) {
	// When a WikiRetriever is wired, wiki_query returns the parsed page payload
	// in the same shape Python's action layer consumes.
	ex := &searchExecutor{deps: SearchDeps{WikiRetriever: &fakeWikiRetriever{}}, req: RunRequest{}}

	oc, err := ex.Execute(context.Background(), "wiki_query", map[string]any{"query": "Culdcept overview"})
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if oc.Status != StatusOK {
		t.Fatalf("status = %s, want ok", oc.Status)
	}
	if n, _ := toInt(oc.Metrics["n_hits"]); n != 1 {
		t.Errorf("n_hits = %v, want 1", oc.Metrics["n_hits"])
	}
	hit, ok := oc.Payload[0].(map[string]any)
	if !ok {
		t.Fatalf("payload[0] not a map: %T", oc.Payload[0])
	}
	for _, key := range []string{"chunk_id", "doc_id", "docnm_kwd", "title", "content", "query"} {
		if _, present := hit[key]; !present {
			t.Errorf("payload missing key %q", key)
		}
	}
	if hit["content"] != "Culdcept is a board game by OmiyaSoft." {
		t.Errorf("content = %v", hit["content"])
	}
}

func TestWikiQuerySchemaIsUnplugged(t *testing.T) {
	// wiki_query is ported as an UNPLUGGED extension seam: the handler exists
	// (TestWikiQueryWiredReturnsPages), but it is NOT part of any mode's active
	// tool set (removed from allTools) and no SearchWiki backend is wired in
	// production, so it must never appear in the advertised surface — mirroring
	// Python, which keeps tools/exploration.py::wiki_query but never registers it
	// in the action session.
	for _, mode := range []string{"medium", "high", "ultra"} {
		ts := &Toolset{ThinkingMode: mode}
		if spec, ok := findSpec(ts.ActiveToolSpecs(), "wiki_query"); ok {
			t.Errorf("wiki_query surfaced in mode %q: %+v", mode, spec)
		}
	}
}

// findSpec is a test helper mirroring ActiveToolSpecs lookup.
func findSpec(specs []ToolSpec, name string) (ToolSpec, bool) {
	for _, s := range specs {
		if s.Function.Name == name {
			return s, true
		}
	}
	return ToolSpec{}, false
}

func TestNavigationToolsReportEmptyWhenUncompiled(t *testing.T) {
	// The Go port has no compiled navigation structures, so navigate_tree /
	// navigate_structure report EMPTY/no_structure — the DATASET-level signal
	// that lets the ladder fall through to `global` and (after the strikes)
	// disable the tool, instead of looping on a rung that cannot succeed.
	kb := &Kbinfos{}
	ex := &searchExecutor{deps: SearchDeps{KB: kb}}
	// No database is wired in this unit test, so defaultSeedEncoder's nil-DB
	// guard returns nil and structure navigation falls back to keyword matching
	// instead of panicking.

	oc, _ := ex.Execute(context.Background(), "navigate_tree", map[string]any{"query": "topic"})
	if oc.Status != StatusEmpty || oc.Reason != ReasonNoStructure {
		t.Errorf("navigate_tree: got (%s,%s), want (empty,no_structure)", oc.Status, oc.Reason)
	}
	// A note must accompany it, so the model knows to switch tools.
	if len(oc.Payload) == 0 {
		t.Error("navigate_tree must return an actionable note, not an empty payload")
	}

	oc, _ = ex.Execute(context.Background(), "navigate_structure", map[string]any{"doc_id": "d1", "query": "topic"})
	if oc.Status != StatusEmpty || oc.Reason != ReasonNoStructure {
		t.Errorf("navigate_structure: got (%s,%s), want (empty,no_structure)", oc.Status, oc.Reason)
	}
}

func TestListChunksReadsFromEvidencePool(t *testing.T) {
	// list_chunks deep-reads a document from the accumulated pool. An unknown
	// doc_id is a MISS (query-level), not an EMPTY — the tool is fine, this id
	// simply is not in evidence.
	kb := &Kbinfos{Chunks: []map[string]any{
		{"chunk_id": "c1", "doc_id": "doc-a", "content": "first"},
		{"chunk_id": "c2", "doc_id": "doc-b", "content": "other"},
		{"chunk_id": "c3", "doc_id": "doc-a", "content": "second"},
	}}
	ex := &searchExecutor{deps: SearchDeps{KB: kb}}

	oc, _ := ex.Execute(context.Background(), "list_chunks", map[string]any{"doc_id": "doc-a"})
	if oc.Status != StatusOK {
		t.Fatalf("status = %s, want ok", oc.Status)
	}
	if len(oc.Payload) != 2 {
		t.Errorf("payload = %d, want 2 chunks of doc-a", len(oc.Payload))
	}
	// Evidence ids are the pool indices, which is what slot patches reference.
	if len(oc.EvidenceIDs) != 2 || oc.EvidenceIDs[0] != "0" || oc.EvidenceIDs[1] != "2" {
		t.Errorf("evidence ids = %v, want [0 2]", oc.EvidenceIDs)
	}

	oc, _ = ex.Execute(context.Background(), "list_chunks", map[string]any{"doc_id": "doc-zz"})
	if oc.Status != StatusMiss {
		t.Errorf("unknown doc status = %s, want miss", oc.Status)
	}
	// Missing doc_id is bad args, not a miss.
	oc, _ = ex.Execute(context.Background(), "list_chunks", map[string]any{})
	if oc.Status != StatusError || oc.Reason != ReasonBadArgs {
		t.Errorf("no doc_id: got (%s,%s), want (error,bad_args)", oc.Status, oc.Reason)
	}
}

func TestToolQueriesAcceptsBothShapes(t *testing.T) {
	// Models emit `query` as a string OR a list, per the advertised schema.
	if got := toolQueries(map[string]any{"query": "single"}); len(got) != 1 || got[0] != "single" {
		t.Errorf("string form = %v", got)
	}
	if got := toolQueries(map[string]any{"query": []any{"a", "", "b"}}); len(got) != 2 {
		t.Errorf("list form = %v, want blanks dropped", got)
	}
	if got := toolQueries(map[string]any{"query": []string{"a", "b"}}); len(got) != 2 {
		t.Errorf("[]string form = %v", got)
	}
	if got := toolQueries(map[string]any{}); got != nil {
		t.Errorf("absent query = %v, want nil", got)
	}
	// The `q` alias some models emit.
	if got := toolQueries(map[string]any{"q": "alias"}); len(got) != 1 || got[0] != "alias" {
		t.Errorf("q alias = %v", got)
	}
}

func TestPublishReferencesSkipsWithoutCanvasState(t *testing.T) {
	// No canvas state attached: publishing is a silent no-op (best-effort),
	// which is what happens in unit tests and non-canvas callers.
	kb := &Kbinfos{Chunks: []map[string]any{{"content": "x", "doc_id": "d1"}}}
	PublishReferences(context.Background(), kb) // must not panic
}

func TestPassageFromChunkTruncatesContent(t *testing.T) {
	long := strings.Repeat("word ", 500)
	p := passageFromChunk(map[string]any{
		"chunk_id": "c1", "doc_id": "d1", "docnm_kwd": "Title", "content": long,
	}, "q")
	content, _ := p["content"].(string)
	// Non-table chunk is capped at 1200 chars (mirrors Python _admit_evidence),
	// table chunks pass through untruncated.
	if len(content) > 1220 {
		t.Errorf("content not truncated: %d chars", len(content))
	}
	if !strings.HasSuffix(content, "...") {
		t.Errorf("truncated content must end with an ellipsis: %q", content)
	}
	if p["doc_id"] != "d1" || p["title"] != "Title" {
		t.Errorf("passage = %v", p)
	}
}
