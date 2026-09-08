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
	"errors"
	"strings"
	"testing"

	"github.com/cloudwego/eino/schema"
	"gorm.io/gorm"

	"ragflow/internal/agent/chat"
	"ragflow/internal/agent/component"
	"ragflow/internal/service/nav"
)

// ---------------------------------------------------------------------------
// Compiled-structure rendering / normalization
// ---------------------------------------------------------------------------

func TestNormalizeKind(t *testing.T) {
	cases := []struct {
		row  map[string]interface{}
		want string
	}{
		{map[string]interface{}{"compile_kwd": "raptor_graph"}, "raptor"},
		{map[string]interface{}{"compilation_template_kind_kwd": "page_index"}, "timeline"},
		{map[string]interface{}{"compilation_template_kind_kwd": "knowledge_graph"}, "timeline"},
		{map[string]interface{}{"compilation_template_kind_kwd": "mindmap"}, "mindmap"},
		{map[string]interface{}{"compile_kwd": "tree"}, "tree"},
	}
	for _, c := range cases {
		if got := normalizeKind(c.row); got != c.want {
			t.Errorf("normalizeKind(%v) = %q, want %q", c.row, got, c.want)
		}
	}
}

func containsAll(s string, subs ...string) bool {
	for _, sub := range subs {
		if !strings.Contains(s, sub) {
			return false
		}
	}
	return true
}

// ---------------------------------------------------------------------------
// navigate_tree routing
// ---------------------------------------------------------------------------

type stubRouter struct {
	table       [][2]string
	nilResult   bool
	emptyResult bool
	err         error
	gotQuery    string
	gotDocScope []string
	gotTopK     int
}

func (s *stubRouter) Route(_ context.Context, _, _, query string, docScope []string, topK int) ([][2]string, error) {
	s.gotQuery = query
	s.gotDocScope = docScope
	s.gotTopK = topK
	if s.err != nil {
		return nil, s.err
	}
	if s.nilResult {
		return nil, nil
	}
	if s.emptyResult {
		return [][2]string{}, nil
	}
	return s.table, nil
}

func TestNavigateTreeRoutesAndRendersXML(t *testing.T) {
	router := &stubRouter{table: [][2]string{
		{"doc-c", "Culdcept history and design"},
		{"doc-a", "OmiyaSoft company profile"},
		{"doc-b", ""},
	}}
	got := NavigateTree(context.Background(), router, NavTreeInput{
		Query:    "Who created Culdcept?",
		Keywords: "Culdcept, creator",
		KbIDs:    []string{"kb1"},
	})
	if got.EmptyReason != "" {
		t.Fatalf("empty_reason = %q, want a successful route", got.EmptyReason)
	}
	if len(got.DocIDs) != 3 {
		t.Fatalf("doc_ids = %v, want 3", got.DocIDs)
	}
	if len(got.RoutedDocs) != 3 {
		t.Fatalf("routed_docs = %v, want 3", got.RoutedDocs)
	}
	if got.RoutedDocs[0][1] != "Culdcept history and design" {
		t.Errorf("routed summary = %q", got.RoutedDocs[0][1])
	}
	if !strings.Contains(router.gotQuery, "Culdcept, creator") {
		t.Errorf("routing query = %q, want topic + keywords", router.gotQuery)
	}
	if !strings.Contains(got.Text, `count="3"`) {
		t.Errorf("XML missing the count: %s", got.Text)
	}
	if !strings.Contains(got.Text, `doc_id="doc-c"`) {
		t.Errorf("XML missing doc ids: %s", got.Text)
	}
	if !strings.Contains(got.Text, "<summary>Culdcept history and design</summary>") {
		t.Errorf("XML missing the summary: %s", got.Text)
	}
	if !strings.Contains(got.Text, `doc_id="doc-b"/>`) {
		t.Errorf("summary-less doc must be self-closing: %s", got.Text)
	}
	if !strings.HasSuffix(strings.TrimSpace(got.Text), "</tree_navigation>") {
		t.Errorf("XML not closed: %s", got.Text)
	}
}

func TestNavigateTreeDistinguishesNoStructureFromNoDoc(t *testing.T) {
	got := NavigateTree(context.Background(), &stubRouter{nilResult: true}, NavTreeInput{
		Query: "q", KbIDs: []string{"kb1"},
	})
	if got.EmptyReason != ReasonNoStructure {
		t.Errorf("no tree: reason = %q, want %q", got.EmptyReason, ReasonNoStructure)
	}
	if got.HasStructure() {
		t.Error("no tree must report HasStructure=false")
	}
	got = NavigateTree(context.Background(), &stubRouter{emptyResult: true}, NavTreeInput{
		Query: "q", KbIDs: []string{"kb1"},
	})
	if got.EmptyReason != ReasonNoDoc {
		t.Errorf("routed to nothing: reason = %q, want %q", got.EmptyReason, ReasonNoDoc)
	}
	if !got.HasStructure() {
		t.Error("a query-level miss must still report HasStructure=true")
	}
}

func TestNavigateTreeBadArgsAndInfra(t *testing.T) {
	if got := NavigateTree(context.Background(), &stubRouter{}, NavTreeInput{}); got.EmptyReason != ReasonBadArgs {
		t.Errorf("no query: reason = %q, want bad_args", got.EmptyReason)
	}
	if got := NavigateTree(context.Background(), nil, NavTreeInput{Query: "q"}); got.EmptyReason != ReasonInfra {
		t.Errorf("no router: reason = %q, want infra", got.EmptyReason)
	}
	got := NavigateTree(context.Background(), &stubRouter{err: errors.New("ES down")}, NavTreeInput{
		Query: "q", KbIDs: []string{"kb1"},
	})
	if got.EmptyReason != ReasonNoStructure {
		t.Errorf("backend error: reason = %q, want no_structure (no dataset verdict)", got.EmptyReason)
	}
}

func TestNavigateTreeCachesBestPerDocAcrossDatasets(t *testing.T) {
	router := &stubRouter{table: [][2]string{
		{"doc-a", "summary from kb1"},
		{"doc-a", "summary from kb2"},
		{"doc-b", "b"},
	}}
	got := NavigateTree(context.Background(), router, NavTreeInput{
		Query: "q", KbIDs: []string{"kb1", "kb2"},
	})
	if len(got.DocIDs) != 2 {
		t.Fatalf("doc_ids = %v, want 2 (deduped)", got.DocIDs)
	}
}

func TestNavigateTreeCapsAt8(t *testing.T) {
	var table [][2]string
	for i := 0; i < 20; i++ {
		table = append(table, [2]string{string(rune('a'+i%26)) + string(rune('0'+i/26)), "s"})
	}
	got := NavigateTree(context.Background(), &stubRouter{table: table}, NavTreeInput{
		Query: "q", KbIDs: []string{"kb1"},
	})
	if len(got.DocIDs) != navTreeMaxDocs {
		t.Errorf("doc_ids = %d, want capped at %d", len(got.DocIDs), navTreeMaxDocs)
	}
}

func TestNavigateTreeEscapesXML(t *testing.T) {
	router := &stubRouter{table: [][2]string{{"d&<>\"'1", "a < b & c"}}}
	got := NavigateTree(context.Background(), router, NavTreeInput{
		Query: `who "made" <it> & that`, KbIDs: []string{"kb1"},
	})
	if strings.Contains(got.Text, "<it>") {
		t.Errorf("query not escaped: %s", got.Text)
	}
	if !strings.Contains(got.Text, "&lt;it&gt;") {
		t.Errorf("query not escaped: %s", got.Text)
	}
	if strings.Contains(got.Text, `doc_id="d&<>\"'1"`) {
		t.Errorf("doc_id not escaped: %s", got.Text)
	}
}

// ---------------------------------------------------------------------------
// Compiled-structure parsing
// ---------------------------------------------------------------------------

func TestParseCompiledStructureGraphBlob(t *testing.T) {
	rows := []StructureRow{{
		CompileKwd:        "tree",
		KnowledgeGraphKwd: "graph",
		Content:           `{"entities": [{"name": "OmiyaSoft"}, {"name": "Culdcept"}], "relations": [{"src": "OmiyaSoft", "dst": "Culdcept"}]}`,
	}}
	ents, rels := ParseCompiledStructure(rows, []string{"tree"})
	if len(ents) != 2 || len(rels) != 1 {
		t.Fatalf("blob: entities=%d relations=%d, want 2/1", len(ents), len(rels))
	}
}

func TestParseCompiledStructurePerEntityRows(t *testing.T) {
	rows := []StructureRow{
		{CompileKwd: "page_index", KnowledgeGraphKwd: "entity", Content: `{"name": "OmiyaSoft"}`},
		{CompileKwd: "page_index", KnowledgeGraphKwd: "entity", Content: `{"name": "Culdcept"}`},
		{CompileKwd: "page_index", KnowledgeGraphKwd: "relation", Content: `{"src": "OmiyaSoft", "dst": "Culdcept"}`},
	}
	ents, rels := ParseCompiledStructure(rows, []string{"timeline"})
	if len(ents) != 2 || len(rels) != 1 {
		t.Fatalf("per-row: entities=%d relations=%d, want 2/1", len(ents), len(rels))
	}
}

func TestParseCompiledStructureKindFilterAndNormalization(t *testing.T) {
	rows := []StructureRow{
		{CompileKwd: "page_index", KnowledgeGraphKwd: "entity", Content: `{"name": "A"}`},
		{CompileKwd: "knowledge_graph", KnowledgeGraphKwd: "entity", Content: `{"name": "B"}`},
		{CompileKwd: "tree", KnowledgeGraphKwd: "entity", Content: `{"name": "C"}`},
		{CompileKwd: "raptor_graph", KnowledgeGraphKwd: "graph", Content: `{"entities": [{"name": "D"}]}`},
	}
	ents, _ := ParseCompiledStructure(rows, []string{"timeline"})
	if len(ents) != 2 {
		t.Errorf("timeline filter = %d entities, want 2 (page_index + knowledge_graph)", len(ents))
	}
	ents, _ = ParseCompiledStructure(rows, []string{"raptor"})
	if len(ents) != 1 {
		t.Errorf("raptor filter = %d entities, want 1", len(ents))
	}
	ents, _ = ParseCompiledStructure([]StructureRow{
		{CompileKwd: "", TemplateKind: "tree", KnowledgeGraphKwd: "entity", Content: `{"name": "E"}`},
	}, []string{"tree"})
	if len(ents) != 1 {
		t.Errorf("template_kind fallback = %d, want 1", len(ents))
	}
}

func TestParseCompiledStructureSkipsMalformed(t *testing.T) {
	rows := []StructureRow{
		{CompileKwd: "tree", KnowledgeGraphKwd: "entity", Content: "not json"},
		{CompileKwd: "tree", KnowledgeGraphKwd: "entity", Content: `[1,2,3]`},
		{CompileKwd: "tree", KnowledgeGraphKwd: "unknown_shape", Content: `{"name":"X"}`},
		{CompileKwd: "tree", KnowledgeGraphKwd: "entity", Content: `{"name": "ok"}`},
	}
	ents, _ := ParseCompiledStructure(rows, []string{"tree"})
	if len(ents) != 1 {
		t.Errorf("entities = %d, want 1 (malformed rows skipped)", len(ents))
	}
}

func TestParseCompiledStructureEmptyKindsMatchesAll(t *testing.T) {
	rows := []StructureRow{
		{CompileKwd: "tree", KnowledgeGraphKwd: "entity", Content: `{"name": "A"}`},
		{CompileKwd: "other", KnowledgeGraphKwd: "entity", Content: `{"name": "B"}`},
	}
	ents, _ := ParseCompiledStructure(rows, nil)
	if len(ents) != 2 {
		t.Errorf("no filter = %d entities, want 2", len(ents))
	}
}

// ---------------------------------------------------------------------------
// Dataset navigation (NavigateDatasetByTree / askNavSelect / collectNavLeaves)
// ---------------------------------------------------------------------------

// TestAskNavSelect_IndexBased asserts the model's index-based selection maps
// back to the item subset.
func TestAskNavSelect_IndexBased(t *testing.T) {
	installChat(t, `{"relevant":[0,2]}`)
	items := []navSelectItem{
		{Name: "Alpha", Description: "aaa"},
		{Name: "Beta", Description: "bbb"},
		{Name: "Gamma", Description: "ggg"},
	}
	out := askNavSelect(t.Context(), nil, "query", "clusters", items, 10)
	if len(out) != 2 {
		t.Fatalf("selected = %d, want 2", len(out))
	}
	if out[0].Name != "Alpha" || out[1].Name != "Gamma" {
		t.Errorf("selected names = %q, %q; want Alpha, Gamma", out[0].Name, out[1].Name)
	}
}

// TestAskNavSelect_Empty asserts an empty "relevant" list yields nothing.
func TestAskNavSelect_Empty(t *testing.T) {
	installChat(t, `{"relevant":[]}`)
	if out := askNavSelect(t.Context(), nil, "q", "clusters", []navSelectItem{{Name: "A"}}, 10); len(out) != 0 {
		t.Errorf("expected no selection, got %d", len(out))
	}
}

// TestAskNavSelect_OutOfRange asserts invalid indices are skipped.
func TestAskNavSelect_OutOfRange(t *testing.T) {
	installChat(t, `{"relevant":[0,99,-1]}`)
	out := askNavSelect(t.Context(), nil, "q", "clusters", []navSelectItem{{Name: "A"}, {Name: "B"}}, 10)
	if len(out) != 1 || out[0].Name != "A" {
		t.Errorf("out-of-range selection = %+v, want [A]", out)
	}
}

// TestNavigateDatasetByTree_NoQuery asserts empty query returns nil without
// calling anything.
func TestNavigateDatasetByTree_NoQuery(t *testing.T) {
	if out := NavigateDatasetByTree(t.Context(), nil, nil, "t1", "kb1", "  "); out != nil {
		t.Errorf("expected nil for empty query, got %v", out)
	}
}

// fakeNavSvcHarness is an in-memory nav.NavService for the BFS test.
type fakeNavSvcHarness struct {
	clusters []nav.NavNode
	children map[string][]nav.NavNode
}

func (f *fakeNavSvcHarness) UpsertDoc(context.Context, nav.UpsertDocInput) error { return nil }
func (f *fakeNavSvcHarness) RemoveDoc(context.Context, string, string, string) error {
	return nil
}
func (f *fakeNavSvcHarness) Search(context.Context, string, string, string, []float32, int) ([]nav.NavHit, error) {
	return nil, nil
}
func (f *fakeNavSvcHarness) ListClusters(context.Context, string, string, int, int) ([]nav.NavNode, int64, error) {
	return f.clusters, int64(len(f.clusters)), nil
}
func (f *fakeNavSvcHarness) ListChildren(_ context.Context, _, _, name string, _, _ int) ([]nav.NavNode, int64, error) {
	return f.children[name], int64(len(f.children[name])), nil
}
func (f *fakeNavSvcHarness) SummariesByDocIDs(context.Context, string, string, []string) map[string]string {
	return map[string]string{}
}

// TestCollectNavLeaves_BFS asserts document leaves are collected, sub-clusters
// descended, and leaves deduped by doc_id.
func TestCollectNavLeaves_BFS(t *testing.T) {
	ns := &fakeNavSvcHarness{
		clusters: []nav.NavNode{{Name: "C1", Description: "cluster 1"}},
		children: map[string][]nav.NavNode{
			"C1": {
				{Name: "Sub", Type: "cluster"},
				{Name: "DocA", Type: "doc", DocID: "d1"},
				{Name: "DocB", Type: "doc", DocID: "d2"},
			},
			"Sub": {{Name: "DocC", Type: "doc", DocID: "d3"}},
		},
	}
	selected := []navSelectItem{{Name: "C1", Description: "cluster 1"}}
	leaves := collectNavLeaves(t.Context(), ns, "t1", "kb1", selected)
	if len(leaves) != 3 {
		t.Fatalf("leaves = %d, want 3 (d1,d2 from C1 + d3 from Sub)", len(leaves))
	}
	seen := map[string]bool{}
	for _, l := range leaves {
		if l.DocID == "" {
			t.Errorf("leaf %q has empty doc_id", l.Name)
		}
		seen[l.DocID] = true
	}
	if !seen["d1"] || !seen["d2"] || !seen["d3"] {
		t.Errorf("collected doc_ids = %v, want d1,d2,d3", seen)
	}
}

// ---------------------------------------------------------------------------
// Knowledge-graph exploration
// ---------------------------------------------------------------------------

// TestExploreGraphShortCircuitsOnEmptyInput verifies the documented contract:
// no query text or no bound datasets yields an empty ExploreResult (no engine
// call). The engine-backed happy path needs a live Infinity backend and is not
// exercised in the unit tier.
func TestExploreGraphShortCircuitsOnEmptyInput(t *testing.T) {
	// No query text.
	if res, err := ExploreGraph(context.Background(), SearchDeps{}, "t1", []string{"kb1"}, "", "", nil); err != nil || res.Answer != "" || len(res.Chunks) != 0 {
		t.Errorf("empty query -> (%+v, %v), want empty result", res, err)
	}
	// No bound datasets.
	if res, err := ExploreGraph(context.Background(), SearchDeps{}, "t1", nil, "OmiyaSoft", "", nil); err != nil || res.Answer != "" || len(res.Chunks) != 0 {
		t.Errorf("no datasets -> (%+v, %v), want empty result", res, err)
	}
}

// TestExploreGraphUnconfiguredEngineErrors verifies a missing engine is
// surfaced rather than silently returning empty.
func TestExploreGraphUnconfiguredEngineErrors(t *testing.T) {
	if _, err := ExploreGraph(context.Background(), SearchDeps{}, "t1", []string{"kb1"}, "OmiyaSoft", "", nil); err == nil {
		t.Error("expected an error when the engine is not configured")
	}
}

// ---------------------------------------------------------------------------
// Shared LLM stub for the tools package's tests. The harness drives the LLM
// through the chat.Invoker seam (internal/agent/chat), which lets a test drive
// navigation and compilation without a real model.
// ---------------------------------------------------------------------------

// fakeChatInvoker returns a canned response for every request and records the
// last prompt it saw.
type fakeChatInvoker struct {
	content  string
	messages []schema.Message
	calls    int
}

func (f *fakeChatInvoker) Invoke(_ context.Context, _ *gorm.DB, req chat.Request) (*chat.Response, error) {
	f.calls++
	f.messages = req.Messages
	return &chat.Response{Content: f.content, Stopped: true}, nil
}

// installChat installs a stub invoker that always replies with content, and
// restores the unconfigured state on test cleanup.
func installChat(t *testing.T, content string) *fakeChatInvoker {
	t.Helper()
	inv := &fakeChatInvoker{content: content}
	component.SetDefaultChatInvoker(inv)
	t.Cleanup(func() { component.SetDefaultChatInvoker(nil) })
	return inv
}

// When the chat invoker is not configured (e.g. unit tests or a server that
// started without an LLM), askNavSelect must short-circuit to nil instead of
// panicking or building a request. This is the canary that the chat singleton
// seam (chat.SetDefaultInvoker) is honoured.
func TestAskNavSelectNilInvoker(t *testing.T) {
	chat.SetDefaultInvoker(nil)
	defer chat.SetDefaultInvoker(nil)

	items := []navSelectItem{
		{Name: "Cluster A", Description: "about rockets", DocCount: 3},
		{Name: "Cluster B", Description: "about games", DocCount: 1},
	}
	got := askNavSelect(context.Background(), nil, "what fits?", "cluster", items, 8)
	if got != nil {
		t.Fatalf("askNavSelect with nil invoker = %v, want nil", got)
	}
}

// Empty input must short-circuit before any invoker lookup.
func TestAskNavSelectEmptyItems(t *testing.T) {
	chat.SetDefaultInvoker(nil)
	defer chat.SetDefaultInvoker(nil)

	if got := askNavSelect(context.Background(), nil, "q", "cluster", nil, 8); got != nil {
		t.Errorf("empty items should return nil, got %v", got)
	}
}

// ---------------------------------------------------------------------------
// Knowledge-graph row parsing
// ---------------------------------------------------------------------------

func TestKgParseEntity_NameFallbacks(t *testing.T) {
	// name wins.
	if e, ok := kgParseEntity(map[string]interface{}{"content_with_weight": `{"name":"OmiyaSoft"}`}); !ok || e.Name != "OmiyaSoft" {
		t.Fatalf("name parse = %+v ok=%v", e, ok)
	}
	// term fallback.
	if e, ok := kgParseEntity(map[string]interface{}{"content_with_weight": `{"term":"Culdcept"}`}); !ok || e.Name != "Culdcept" {
		t.Fatalf("term fallback = %+v ok=%v", e, ok)
	}
	// title fallback.
	if e, ok := kgParseEntity(map[string]interface{}{"content_with_weight": `{"title":"Rocket"}`}); !ok || e.Name != "Rocket" {
		t.Fatalf("title fallback = %+v ok=%v", e, ok)
	}
	// aliases parsed + doc_id carried.
	row := map[string]interface{}{
		"content_with_weight": `{"name":"OmiyaSoft","type":"company","description":"dev","aliases":["Omiya","OS"]}`,
		"doc_id":              "d1",
		"source_chunk_ids":    []interface{}{"c1", "c2"},
	}
	e, ok := kgParseEntity(row)
	if !ok || e.Name != "OmiyaSoft" || e.Type != "company" || e.DocID != "d1" {
		t.Fatalf("entity = %+v ok=%v", e, ok)
	}
	if len(e.Aliases) != 2 || e.Aliases[0] != "Omiya" {
		t.Errorf("aliases = %v, want [Omiya OS]", e.Aliases)
	}
	if len(e.SourceChunkIDs) != 2 {
		t.Errorf("source_chunk_ids = %v, want [c1 c2]", e.SourceChunkIDs)
	}
}

func TestKgParseEntityRejectsEmptyName(t *testing.T) {
	cases := []string{
		`{"type":"x"}`,  // no name
		`{"name":""}`,   // empty name
		`{"name":null}`, // null name
		`not json`,      // unparsable
		`[1,2,3]`,       // not an object
	}
	for _, c := range cases {
		if _, ok := kgParseEntity(map[string]interface{}{"content_with_weight": c}); ok {
			t.Errorf("content %q should not parse to an entity", c)
		}
	}
}

func TestKgParseRelationEndpointsRequired(t *testing.T) {
	// Missing from endpoint.
	if _, ok := kgParseRelation(map[string]interface{}{"to_entity_kwd": "B"}); ok {
		t.Error("relation without from endpoint must be rejected")
	}
	// Missing to endpoint.
	if _, ok := kgParseRelation(map[string]interface{}{"from_entity_kwd": "A"}); ok {
		t.Error("relation without to endpoint must be rejected")
	}
	// Valid: type from content payload takes precedence over default.
	row := map[string]interface{}{
		"from_entity_kwd":     "A",
		"to_entity_kwd":       "B",
		"content_with_weight": `{"type":"founded"}`,
		"doc_id":              "d1",
	}
	r, ok := kgParseRelation(row)
	if !ok || r.From != "A" || r.To != "B" || r.Type != "founded" || r.DocID != "d1" {
		t.Fatalf("relation = %+v ok=%v", r, ok)
	}
	// Default type when payload has neither type nor relation.
	r2, ok := kgParseRelation(map[string]interface{}{
		"from_entity_kwd": "A", "to_entity_kwd": "B",
		"content_with_weight": `{}`,
	})
	if !ok || r2.Type != "related" {
		t.Errorf("default relation type = %q, want related", r2.Type)
	}
}

// ---------------------------------------------------------------------------
// endpoint terms (original + lowercased, deduped)
// ---------------------------------------------------------------------------

func TestEndpointTerms(t *testing.T) {
	got := endpointTerms([]string{"OmiyaSoft", "omiya", "", "Culdcept"})
	want := map[string]bool{
		"OmiyaSoft": true, "omiyasoft": true, "omiya": true, "Culdcept": true, "culdcept": true,
	}
	if len(got) != len(want) {
		t.Fatalf("endpointTerms = %v, want %v", got, want)
	}
	for _, g := range got {
		if !want[g] {
			t.Errorf("unexpected term %q", g)
		}
	}
}

// ---------------------------------------------------------------------------
// Evidence collection: relevant entities + relations grouped by doc
// ---------------------------------------------------------------------------

func TestCollectEvidenceIDsByDocAndAlias(t *testing.T) {
	entities := []kgEntity{
		{Name: "OmiyaSoft", Aliases: []string{"Omiya"}, DocID: "d1", SourceChunkIDs: []string{"c1", "c2"}},
		{Name: "Culdcept", DocID: "d2", SourceChunkIDs: []string{"c3"}},
	}
	relations := []kgRelation{
		{From: "OmiyaSoft", To: "Culdcept", DocID: "d1", SourceChunkIDs: []string{"c4"}},
	}
	// Relevant set includes an alias ("Omiya") and the relation target ("Culdcept").
	byDoc := collectEvidenceIDs(entities, relations, []string{"Omiya", "Culdcept"})
	if len(byDoc["d1"]) != 3 {
		t.Errorf("d1 chunks = %v, want [c1 c2 c4]", byDoc["d1"])
	}
	if len(byDoc["d2"]) != 1 || byDoc["d2"][0] != "c3" {
		t.Errorf("d2 chunks = %v, want [c3]", byDoc["d2"])
	}
	// A non-relevant entity must contribute nothing.
	byDoc = collectEvidenceIDs(entities, relations, []string{"nobody"})
	if len(byDoc) != 0 {
		t.Errorf("non-relevant selection = %v, want empty", byDoc)
	}
}

func TestCollectEvidenceIDsDeduplicatesChunks(t *testing.T) {
	entities := []kgEntity{
		{Name: "A", DocID: "d1", SourceChunkIDs: []string{"c1", "c1"}},
	}
	rel := []kgRelation{
		{From: "A", To: "B", DocID: "d1", SourceChunkIDs: []string{"c1"}},
	}
	byDoc := collectEvidenceIDs(entities, rel, []string{"A", "B"})
	if seen := map[string]int{}; true {
		for _, c := range byDoc["d1"] {
			seen[c]++
		}
		if seen["c1"] != 1 {
			t.Errorf("chunk c1 counted %d times, want 1 (deduped)", seen["c1"])
		}
	}
}

// ---------------------------------------------------------------------------
// mention_count re-ranking (the dense-seed re-sort)
// ---------------------------------------------------------------------------

func TestMentionCountTypeCoercion(t *testing.T) {
	cases := []struct {
		in   interface{}
		want int
	}{
		{int(5), 5},
		{int64(7), 7},
		{float64(9), 9},
		{float32(3), 3},
		{nil, 0},
		{"x", 0},
	}
	for _, c := range cases {
		if got := mentionCount(map[string]interface{}{"mention_count_int": c.in}); got != c.want {
			t.Errorf("mentionCount(%T(%v)) = %d, want %d", c.in, c.in, got, c.want)
		}
	}
}

func TestTopMentionCountSortsAndCaps(t *testing.T) {
	rows := []map[string]interface{}{
		{"mention_count_int": 1, "name": "a"},
		{"mention_count_int": 10, "name": "b"},
		{"mention_count_int": 5, "name": "c"},
	}
	got := topMentionCount(rows, 2)
	if len(got) != 2 || mentionCount(got[0]) != 10 || mentionCount(got[1]) != 5 {
		t.Fatalf("topMentionCount = %v, want [10 5]", got)
	}
}

// ---------------------------------------------------------------------------
// String-field helpers
// ---------------------------------------------------------------------------

func TestStrSliceField(t *testing.T) {
	if got := strSliceField([]string{"a", "b"}); len(got) != 2 {
		t.Errorf("[]string path lost elements: %v", got)
	}
	if got := strSliceField([]interface{}{"a", 2, "b"}); len(got) != 2 {
		t.Errorf("[]interface{} path must skip non-strings: %v", got)
	}
	if got := strSliceField(42); got != nil {
		t.Errorf("scalar path must yield nil, got %v", got)
	}
}

func TestStrOr(t *testing.T) {
	if strOr("x", "def") != "x" {
		t.Error("non-empty value must win")
	}
	if strOr("", "def") != "def" {
		t.Error("empty value must fall back")
	}
	if strOr(nil, "def") != "def" {
		t.Error("non-string must fall back")
	}
}
