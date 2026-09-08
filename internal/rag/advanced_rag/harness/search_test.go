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
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"ragflow/internal/entity"
)

// stubRetriever returns a fixed result and records the requests it received.
type stubRetriever struct {
	chunks   []map[string]any
	err      error
	requests []RetrieveRequest
}

func (s *stubRetriever) Retrieve(_ context.Context, req RetrieveRequest) ([]map[string]any, error) {
	s.requests = append(s.requests, req)
	if s.err != nil {
		return nil, s.err
	}
	return s.chunks, nil
}

func newTestSearchDeps(r Retriever) (SearchDeps, *Kbinfos) {
	kb := &Kbinfos{}
	return SearchDeps{
		Backend:  r,
		KbIDs:    []string{"kb1"},
		TenantID: "tenant1",
		KB:       kb,
	}, kb
}

func TestHybridSearchBailsWithoutDatasetsOrBackend(t *testing.T) {
	// No bound datasets -> no search at all (Python returns empty kbinfos).
	got, aggs := HybridSearch(context.Background(), SearchDeps{
		Backend: &stubRetriever{chunks: []map[string]any{{"content": "x"}}},
	}, SearchParams{Question: "q"})
	if got != nil || aggs != nil {
		t.Error("no datasets must short-circuit to empty")
	}
	// No backend -> empty.
	deps, _ := newTestSearchDeps(nil)
	if got, _ := HybridSearch(context.Background(), deps, SearchParams{Question: "q"}); got != nil {
		t.Error("no backend must return empty")
	}
}

func TestHybridSearchBuildsEffectiveQuery(t *testing.T) {
	// Weighted query wins, and the result is capped at 400 chars.
	r := &stubRetriever{chunks: []map[string]any{{"content": "hit"}}}
	deps, _ := newTestSearchDeps(r)
	long := strings.Repeat("weighted ", 100)
	HybridSearch(context.Background(), deps, SearchParams{
		Question: "who made it?", RetrievalQuery: long,
	})
	if len(r.requests) != 1 {
		t.Fatalf("requests = %d, want 1", len(r.requests))
	}
	q := r.requests[0].Query
	if !strings.HasPrefix(q, "who made it? weighted") {
		t.Errorf("query = %q, want question + weighted query", q)
	}
	if len(q) != maxEffectiveQueryChars {
		t.Errorf("query length = %d, want capped at %d", len(q), maxEffectiveQueryChars)
	}

	// No weighted query -> keywords fall into the query text.
	r = &stubRetriever{chunks: []map[string]any{{"content": "hit"}}}
	deps, _ = newTestSearchDeps(r)
	HybridSearch(context.Background(), deps, SearchParams{Question: "who made it?", Keywords: "culdcept"})
	if r.requests[0].Query != "who made it? culdcept" {
		t.Errorf("query = %q, want keywords appended", r.requests[0].Query)
	}

	// Neither -> the bare question.
	r = &stubRetriever{chunks: []map[string]any{{"content": "hit"}}}
	deps, _ = newTestSearchDeps(r)
	HybridSearch(context.Background(), deps, SearchParams{Question: "who made it?"})
	if r.requests[0].Query != "who made it?" {
		t.Errorf("query = %q, want the bare question", r.requests[0].Query)
	}
}

func TestHybridSearchPassesRankFeature(t *testing.T) {
	// label_question -> rank_feature must flow into the underlying retrieve
	// request (Python retrieve: rank_feature=label_question(question, self.kbs),
	// agentic_rag.py:668). A nil Tagger must leave RankFeature empty.
	r := &stubRetriever{chunks: []map[string]any{{"content": "hit"}}}
	deps, _ := newTestSearchDeps(r)
	deps.KBs = []*entity.Knowledgebase{{}}
	deps.Tagger = stubTagger{t: t}
	HybridSearch(context.Background(), deps, SearchParams{Question: "who made it?"})
	if len(r.requests) != 1 {
		t.Fatalf("requests = %d, want 1", len(r.requests))
	}
	want := map[string]float64{"definition": 1.0, "entity": 1.0}
	if got := r.requests[0].RankFeature; !reflect.DeepEqual(got, want) {
		t.Errorf("RankFeature = %v, want %v", got, want)
	}

	// Nil Tagger -> empty rank feature (Python label_question None).
	r2 := &stubRetriever{chunks: []map[string]any{{"content": "hit"}}}
	deps2, _ := newTestSearchDeps(r2)
	HybridSearch(context.Background(), deps2, SearchParams{Question: "unique query no rf"})
	if len(r2.requests) != 1 {
		t.Fatalf("requests = %d, want 1", len(r2.requests))
	}
	if len(r2.requests[0].RankFeature) != 0 {
		t.Errorf("RankFeature = %v, want empty for nil Tagger", r2.requests[0].RankFeature)
	}
}

// stubTagger implements QuestionLabeler for TestHybridSearchPassesRankFeature.
type stubTagger struct{ t *testing.T }

func (s stubTagger) LabelQuestion(_ context.Context, question string, kbs []*entity.Knowledgebase) map[string]float64 {
	if question != "who made it?" {
		s.t.Errorf("LabelQuestion called with question=%q", question)
	}
	if len(kbs) != 1 {
		s.t.Errorf("LabelQuestion called with %d kbs, want 1", len(kbs))
	}
	return map[string]float64{"definition": 1.0, "entity": 1.0}
}

func TestHybridSearchCachesIdenticalQuery(t *testing.T) {
	r := &stubRetriever{chunks: []map[string]any{{"content": "hit"}}}
	deps, _ := newTestSearchDeps(r)
	p := SearchParams{Question: "same question"}
	HybridSearch(context.Background(), deps, p)
	HybridSearch(context.Background(), deps, p)
	if len(r.requests) != 1 {
		t.Errorf("backend calls = %d, want 1 (second served from cache)", len(r.requests))
	}
	// A different scope is a different retrieval.
	HybridSearch(context.Background(), deps, SearchParams{Question: "same question", DocScope: []string{"doc9"}})
	if len(r.requests) != 2 {
		t.Errorf("backend calls = %d, want 2 (doc_scope is part of the key)", len(r.requests))
	}
	// A different TopN is a different retrieval.
	HybridSearch(context.Background(), deps, SearchParams{Question: "same question", TopN: 5})
	if len(r.requests) != 3 {
		t.Errorf("backend calls = %d, want 3 (top_n is part of the key)", len(r.requests))
	}
}

func TestSearchCacheKeyIgnoresOrderAndWhitespace(t *testing.T) {
	a := SearchCacheKey("  Who   Made  It ", []string{"b", "a"}, 12, []string{"d2", "d1"})
	b := SearchCacheKey("who made it", []string{"a", "b"}, 12, []string{"d1", "d2"})
	if a != b {
		t.Errorf("cache key not canonical:\n%q\n%q", a, b)
	}
}

func TestHybridSearchStoresMemoryBeforeNarrowing(t *testing.T) {
	// The raw chunk must reach memory even when narrowing rewrites the copy
	// handed downstream.
	r := &stubRetriever{chunks: []map[string]any{{
		"chunk_id": "c1",
		"content":  "Alpha sentence. Culdcept was made by OmiyaSoft in 1999. Omega sentence.",
		"doc_id":   "d1",
	}}}
	deps, kb := newTestSearchDeps(r)
	HybridSearch(context.Background(), deps, SearchParams{
		Question: "who made Culdcept?", Keywords: "culdcept",
	})
	if MemorySize(kb) != 1 {
		t.Fatalf("memory size = %d, want 1", MemorySize(kb))
	}
	if strings.Contains(ChunkTextOf(kb.Memory[0]), "<em>") {
		t.Error("memory must hold the RAW chunk, not the narrowed/highlighted copy")
	}
}

func TestHybridSearchReturnsEmptyOnBackendError(t *testing.T) {
	deps, _ := newTestSearchDeps(&stubRetriever{err: errors.New("ES down")})
	got, aggs := HybridSearch(context.Background(), deps, SearchParams{Question: "q"})
	if len(got) != 0 || len(aggs) != 0 {
		t.Errorf("a backend failure must yield empty results, got %d chunks", len(got))
	}
}

// ---------------------------------------------------------------------------
// Narrowing
// ---------------------------------------------------------------------------

func TestNarrowOrKeepIsAllOrNothing(t *testing.T) {
	chunks := []map[string]any{
		{"content": "Culdcept was made by OmiyaSoft."},
		{"content": "It was released in 1999."},
	}
	// Keywords hit -> narrowed subset (the non-matching chunk is dropped).
	got := NarrowOrKeep(chunks, "culdcept", "test", nil)
	if len(got) != 1 || !strings.Contains(ChunkTextOf(got[0]), "Culdcept") {
		t.Errorf("narrowed = %v, want only the matching chunk", got)
	}

	// No keyword overlap -> ALL chunks kept (this is the whole point: the
	// retriever already ranked them, and dropping everything produced empty
	// results and unverified claims).
	got = NarrowOrKeep(chunks, "zzz-no-match", "test", nil)
	if len(got) != len(chunks) {
		t.Errorf("no-match kept %d, want all %d", len(got), len(chunks))
	}

	// Empty keywords -> untouched.
	if got := NarrowOrKeep(chunks, "", "test", nil); len(got) != len(chunks) {
		t.Error("empty keywords must return chunks unchanged")
	}
}

func TestNarrowContentKeepsNeighboursAndHighlights(t *testing.T) {
	content := "Alpha filler. Culdcept was made by OmiyaSoft. Omega filler."
	got, ok := NarrowContent(content, []string{"culdcept"})
	if !ok {
		t.Fatal("NarrowContent must match")
	}
	if !strings.Contains(got, "<em>Culdcept</em>") {
		t.Errorf("keyword not highlighted: %q", got)
	}
	// +/-1 neighbour window: Alpha and Omega should be present.
	if !strings.Contains(got, "Alpha") || !strings.Contains(got, "Omega") {
		t.Errorf("neighbour sentences missing: %q", got)
	}
	if !strings.HasPrefix(got, "...") || !strings.HasSuffix(got, "...") {
		t.Errorf("narrowed text must be wrapped in ellipses: %q", got)
	}
	if _, ok := NarrowContent("nothing relevant here", []string{"culdcept"}); ok {
		t.Error("no match must return false")
	}
}

func TestSplitKeywordsFallsBackToBigrams(t *testing.T) {
	// >=3 comma terms -> used as-is, lower-cased.
	got := SplitKeywords("Alpha, Beta, Gamma")
	if len(got) != 3 || got[0] != "alpha" {
		t.Errorf("comma split = %v", got)
	}
	// <3 terms -> bigrams are more discriminative than single words.
	got = SplitKeywords("finale run time")
	if len(got) != 2 || got[0] != "finale run" || got[1] != "run time" {
		t.Errorf("bigram fallback = %v, want [finale run, run time]", got)
	}
	if SplitKeywords("   ") != nil {
		t.Error("blank keywords must yield no terms")
	}
}

func TestHighlightKeywordsPrefersLongestTerm(t *testing.T) {
	// "new york" must win over "york" so the shorter term cannot split it.
	got := HighlightKeywords("welcome to New York city", []string{"york", "new york"})
	if !strings.Contains(got, "<em>New York</em>") {
		t.Errorf("highlight = %q, want the longest term applied", got)
	}
	if strings.Contains(got, "<em>york</em>") {
		t.Errorf("shorter term must not win: %q", got)
	}
}

// ---------------------------------------------------------------------------
// Doc aggregations
// ---------------------------------------------------------------------------

func TestDocAggsGroupsByDocument(t *testing.T) {
	chunks := []map[string]any{
		{"doc_id": "d1", "docnm_kwd": "Doc One", "content": "aaaa"},
		{"doc_id": "d1", "docnm_kwd": "Doc One", "content": "bb"},
		{"doc_id": "d2", "docnm_kwd": "Doc Two", "content": "c"},
	}
	aggs := DocAggs(chunks)
	if len(aggs) != 2 {
		t.Fatalf("aggs = %d, want 2 documents", len(aggs))
	}
	// Order follows first appearance, so the log line is stable.
	if aggs[0]["doc_id"] != "d1" || aggs[0]["count"] != 2 {
		t.Errorf("d1 agg = %v, want count 2", aggs[0])
	}
	if aggs[0]["char_count"] != 6 {
		t.Errorf("d1 char_count = %v, want 6 (4+2)", aggs[0]["char_count"])
	}
	if aggs[1]["doc_id"] != "d2" || aggs[1]["count"] != 1 {
		t.Errorf("d2 agg = %v", aggs[1])
	}
	if DocAggs(nil) != nil {
		t.Error("no chunks must yield no aggregations")
	}
}

func TestHybridSearchAggregatesAndRespectsScope(t *testing.T) {
	r := &stubRetriever{chunks: []map[string]any{
		{"doc_id": "d1", "content": "a"},
		{"doc_id": "d2", "content": "b"},
	}}
	deps, _ := newTestSearchDeps(r)
	chunks, aggs := HybridSearch(context.Background(), deps, SearchParams{
		Question: "q", DocScope: []string{"d1"},
	})
	if len(chunks) != 2 || len(aggs) != 2 {
		t.Fatalf("chunks=%d aggs=%d", len(chunks), len(aggs))
	}
	// doc_scope must reach the backend.
	if len(r.requests[0].DocScope) != 1 || r.requests[0].DocScope[0] != "d1" {
		t.Errorf("doc_scope not forwarded: %v", r.requests[0].DocScope)
	}
	// Explicit kb_ids override the session default.
	r.requests = nil
	HybridSearch(context.Background(), deps, SearchParams{Question: "q", KbIDs: []string{"kb9"}})
	if len(r.requests[0].DatasetIDs) != 1 || r.requests[0].DatasetIDs[0] != "kb9" {
		t.Errorf("kb_ids not honoured: %v", r.requests[0].DatasetIDs)
	}
}

func TestHybridSearchAggsCoverFullRetrievedBeforeNarrowing(t *testing.T) {
	// doc_aggs must reflect the FULL pre-narrow retrieved set, mirroring Python
	// where _normalize takes doc_aggs from the retriever and _narrow_or_keep only
	// replaces kbinfos["chunks"]. Here the keyword narrows the chunks to doc d1,
	// but the aggregation still counts both retrieved documents.
	r := &stubRetriever{chunks: []map[string]any{
		{"doc_id": "d1", "docnm_kwd": "one", "content": "needle buried here", "id": "c1"},
		{"doc_id": "d2", "docnm_kwd": "two", "content": "nothing relevant", "id": "c2"},
	}}
	deps, _ := newTestSearchDeps(r)
	chunks, aggs := HybridSearch(context.Background(), deps, SearchParams{
		Question: "q", Keywords: "needle",
	})
	if len(chunks) != 1 {
		t.Fatalf("narrowed chunks = %d, want 1 (d1 only)", len(chunks))
	}
	if len(aggs) != 2 {
		t.Fatalf("aggs = %d, want 2 (full pre-narrow set: d1 and d2)", len(aggs))
	}
}

func TestHybridSearchInvokesCompiledExpansion(t *testing.T) {
	r := &stubRetriever{chunks: []map[string]any{{"content": "hit"}}}
	deps, _ := newTestSearchDeps(r)
	exp := &stubExpander{}
	deps.Expand = exp
	HybridSearch(context.Background(), deps, SearchParams{Question: "q", UseCompiled: true})
	if exp.calls != 1 {
		t.Errorf("compiled expansion calls = %d, want 1", exp.calls)
	}
	// Without UseCompiled it must not run.
	exp.calls = 0
	HybridSearch(context.Background(), deps, SearchParams{Question: "other", UseCompiled: false})
	if exp.calls != 0 {
		t.Error("compiled expansion must be opt-in")
	}
}

type stubExpander struct{ calls int }

func (s *stubExpander) Expand(_ context.Context, _ *Kbinfos, _, _ string, _ []string) error {
	s.calls++
	return nil
}

// TestNormalizeWebResults verifies the web-search payload shaper drops entries
// without a URL or content and keeps chunk_id unique per snippet.
func TestNormalizeWebResults(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var in map[string]any
		_ = json.NewDecoder(r.Body).Decode(&in)
		out := map[string]any{
			"chunks": []map[string]any{
				{"url": "u1", "content": "c1", "title": "t1"},
				{"url": "", "content": "c2"},   // missing url -> dropped
				{"url": "u2", "content": ""},   // missing content -> dropped
				{"url": "u3", "content": "c3"}, // kept
				{"url": "u1", "content": "c4"}, // same url as u1 -> must keep (unique chunk_id)
			},
		}
		_ = json.NewEncoder(w).Encode(out)
	}))
	defer srv.Close()

	resp, _ := http.Post(srv.URL, "application/json", bytes.NewReader([]byte(`{}`)))
	raw, _ := io.ReadAll(resp.Body)
	_ = resp.Body.Close()
	got := normalizeWebResults(raw)
	if len(got) != 3 {
		t.Fatalf("got %d chunks, want 3 (u1, u3, u1-second-snippet)", len(got))
	}

	seen := map[string]int{}
	for _, c := range got {
		id, _ := c["chunk_id"].(string)
		seen[id]++
	}
	for id, n := range seen {
		if n != 1 {
			t.Errorf("chunk_id %q appeared %d times, must be unique", id, n)
		}
	}

	// Filtering of bad entries.
	if c := normalizeWebResults([]byte(`{"chunks":[{"content":"no-url"}]}`)); len(c) != 0 {
		t.Errorf("entry without url kept: %v", c)
	}
	if c := normalizeWebResults([]byte(`{"chunks":[{"url":"x"}]}`)); len(c) != 0 {
		t.Errorf("entry without content kept: %v", c)
	}
}
