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

package orchestrator

import (
	"context"
	"errors"
	"strings"
	"testing"

	"ragflow/internal/rag/advanced_rag/harness"
)

// stubJSONModel returns a canned JSON value (or an error), and records the last
// prompt it was shown.
type stubJSONModel struct {
	value any
	err   error
	calls int
	last  string
}

func (s *stubJSONModel) GenJSON(_ context.Context, prompt string) (any, error) {
	s.calls++
	s.last = prompt
	if s.err != nil {
		return nil, s.err
	}
	return s.value, nil
}

// ---------------------------------------------------------------------------
// Direct search
// ---------------------------------------------------------------------------

func TestDirectSearchEmptyWhenNothingFound(t *testing.T) {
	kb := &harness.Kbinfos{}
	got := DirectSearch(context.Background(), DirectDeps{
		KB: kb,
		Search: func(_ context.Context, _ harness.SearchParams) ([]map[string]any, []map[string]any) {
			return nil, nil
		},
	}, "who created Culdcept?", "")
	if !got.EmptyResult {
		t.Error("no hits must set EmptyResult")
	}
}

func TestDirectSearchMergesHits(t *testing.T) {
	kb := &harness.Kbinfos{}
	var got harness.SearchParams
	res := DirectSearch(context.Background(), DirectDeps{
		KB: kb,
		Search: func(_ context.Context, p harness.SearchParams) ([]map[string]any, []map[string]any) {
			got = p
			return []map[string]any{{"content": "hit one"}},
				[]map[string]any{{"doc_id": "d1"}}
		},
	}, "question", "kw")
	if res.EmptyResult {
		t.Error("a hit must clear EmptyResult")
	}
	if len(kb.Chunks) != 1 || len(kb.DocAggs) != 1 {
		t.Fatalf("chunks=%d aggs=%d, want 1/1", len(kb.Chunks), len(kb.DocAggs))
	}
	// The compiled-structure expansion must be enabled, matching Python's
	// use_compiled default for the direct pass.
	if !got.UseCompiled {
		t.Error("direct search must enable compiled expansion")
	}
	if got.Question != "question" || got.Keywords != "kw" {
		t.Errorf("search params = %+v", got)
	}
}

// stubKeywords records whether the weighted extractor was consulted.
type stubKeywords struct {
	called bool
	err    error
}

func (s *stubKeywords) ExtractKeywordsWeighted(_ context.Context, _ string) (string, string, error) {
	s.called = true
	if s.err != nil {
		return "", "", s.err
	}
	return "entity entity entity", "entity", nil
}

func TestDirectSearchAttachesWeightedQuery(t *testing.T) {
	kb := &harness.Kbinfos{}
	kw := &stubKeywords{}
	var got harness.SearchParams
	DirectSearch(context.Background(), DirectDeps{
		KB: kb,
		Search: func(_ context.Context, p harness.SearchParams) ([]map[string]any, []map[string]any) {
			got = p
			return []map[string]any{{"content": "x"}}, nil
		},
		Keywords: kw,
	}, "question", "")
	if !kw.called {
		t.Error("weighted extractor must be consulted")
	}
	if got.RetrievalQuery != "entity entity entity" {
		t.Errorf("retrieval query = %q, want the weighted query", got.RetrievalQuery)
	}
}

func TestDirectSearchSurvivesKeywordFailure(t *testing.T) {
	// A keyword-extraction failure must be logged and ignored, not propagated.
	kb := &harness.Kbinfos{}
	DirectSearch(context.Background(), DirectDeps{
		KB: kb,
		Search: func(_ context.Context, _ harness.SearchParams) ([]map[string]any, []map[string]any) {
			return []map[string]any{{"content": "x"}}, nil
		},
		Keywords: &stubKeywords{err: errors.New("boom")},
	}, "question", "")
	if len(kb.Chunks) != 1 {
		t.Errorf("search must still run after a keyword failure; chunks=%d", len(kb.Chunks))
	}
}

// ---------------------------------------------------------------------------
// SCA
// ---------------------------------------------------------------------------

func TestSCAReturnsNoSignalWithoutClaims(t *testing.T) {
	mdl := &stubJSONModel{value: map[string]any{"is_sufficient": true}}
	got := SufficientContextAgent(context.Background(), SCADeps{
		KB:    &harness.Kbinfos{},
		Model: mdl,
	}, "q", nil)
	if mdl.calls != 0 {
		t.Error("no claims must short-circuit before any LLM call")
	}
	if got.IsSufficient || got.Claims != nil {
		t.Error("no claims must yield an empty result (no signal)")
	}
}

func TestSCAParsesUnifiedVerdict(t *testing.T) {
	mdl := &stubJSONModel{value: map[string]any{
		"is_sufficient":  false,
		"confidence":     0.42,
		"contradictions": []any{"X says 1999, Y says 2001"},
		"reasoning":      "missing the release year",
		"sub_queries": []any{
			map[string]any{"sub_query": "release year", "satisfied": false,
				"missing_fact": "the year", "search_hint": "Culdcept release date"},
			map[string]any{"sub_query": "developer", "satisfied": true},
		},
		"claims": []any{
			map[string]any{"claim_id": "c1", "grounded": false,
				"missing_information": []any{map[string]any{"what": "release year", "search_hint": "Culdcept 1999"}}},
		},
	}}
	got := SufficientContextAgent(context.Background(), SCADeps{
		KB:    &harness.Kbinfos{},
		Model: mdl,
	}, "q", []ClaimDraft{{ID: "c1", Draft: "OmiyaSoft made it"}})

	if got.IsSufficient {
		t.Error("is_sufficient must be false")
	}
	if got.Confidence != 0.42 {
		t.Errorf("confidence = %v, want 0.42", got.Confidence)
	}
	if len(got.Contradictions) != 1 || got.Contradictions[0] != "X says 1999, Y says 2001" {
		t.Errorf("contradictions = %v", got.Contradictions)
	}
	if len(got.SubQueries) != 2 {
		t.Fatalf("sub_queries = %d, want 2", len(got.SubQueries))
	}
	if got.SubQueries[0].Satisfied || got.SubQueries[0].MissingFact != "the year" {
		t.Errorf("sub_query[0] = %+v", got.SubQueries[0])
	}
	if !got.SubQueries[1].Satisfied {
		t.Error("sub_query[1] must be satisfied")
	}
	cv, ok := got.Claims["c1"]
	if !ok {
		t.Fatal("claim c1 missing from verdicts")
	}
	if cv.Grounded {
		t.Error("claim c1 must be ungrounded")
	}
	if len(cv.MissingInformation) != 1 || cv.MissingInformation[0].What != "release year" {
		t.Errorf("missing_information = %+v", cv.MissingInformation)
	}
}

func TestSCACoercesArrayAndStringDrift(t *testing.T) {
	// Model drift: a bare array, or a JSON string, must still yield a verdict
	// instead of being dropped (which would silently stop replan/rewrite).
	for name, value := range map[string]any{
		"array":  []any{map[string]any{"is_sufficient": true, "confidence": 0.9}},
		"string": `{"is_sufficient": true, "confidence": 0.7}`,
	} {
		mdl := &stubJSONModel{value: value}
		got := SufficientContextAgent(context.Background(), SCADeps{
			KB:    &harness.Kbinfos{},
			Model: mdl,
		}, "q", []ClaimDraft{{ID: "c1", Draft: "d"}})
		if !got.IsSufficient {
			t.Errorf("%s drift: verdict lost", name)
		}
	}
}

func TestSCADerivesGapWhenInsufficientWithEmptyClaims(t *testing.T) {
	// Failsafe: insufficient + empty claims array (a known degradation on long
	// prompts) must still produce a gap, else the orchestrator abandons despite
	// having usable evidence.
	mdl := &stubJSONModel{value: map[string]any{
		"is_sufficient":       false,
		"missing_information": []any{map[string]any{"what": "the year", "search_hint": "1999"}},
	}}
	got := SufficientContextAgent(context.Background(), SCADeps{
		KB:    &harness.Kbinfos{},
		Model: mdl,
	}, "q", []ClaimDraft{{ID: "c1", Draft: "OmiyaSoft made it"}})
	g, ok := got.Claims["_global"]
	if !ok {
		t.Fatal("top-level missing_information must be harvested as the _global gap")
	}
	if len(g.MissingInformation) != 1 || g.MissingInformation[0].What != "the year" {
		t.Errorf("gap = %+v", g.MissingInformation)
	}

	// No structured gap at all: fall back to the drafts themselves.
	mdl = &stubJSONModel{value: map[string]any{"is_sufficient": false}}
	got = SufficientContextAgent(context.Background(), SCADeps{
		KB:    &harness.Kbinfos{},
		Model: mdl,
	}, "q", []ClaimDraft{{ID: "c1", Draft: "OmiyaSoft made it"}})
	g, ok = got.Claims["_global"]
	if !ok || len(g.MissingInformation) == 0 {
		t.Fatal("draft-derived fallback gap missing")
	}
	if g.MissingInformation[0].What != "OmiyaSoft made it" {
		t.Errorf("fallback gap = %q, want the draft text", g.MissingInformation[0].What)
	}
}

func TestSCAEvidenceAnchorsUseChunkIndices(t *testing.T) {
	// EvidenceIDs are INDICES into Kbinfos.Chunks (not chunk_id hashes) — that
	// indexing is what makes anchor lookup hit.
	kb := &harness.Kbinfos{Chunks: []map[string]any{
		{"content": "Culdcept was released in 1999 by OmiyaSoft."},
	}}
	contextText := renderClaimContext([]ClaimDraft{
		{ID: "c1", Draft: "OmiyaSoft made Culdcept", EvidenceIDs: []string{"0"}},
	}, kb)
	if !strings.Contains(contextText, "1999") {
		t.Errorf("anchor not resolved from index 0; got %q", contextText)
	}
	// An out-of-range index is simply skipped.
	if s := renderClaimContext([]ClaimDraft{{ID: "c1", Draft: "d", EvidenceIDs: []string{"99"}}}, kb); strings.Contains(s, "Evidence:") {
		t.Error("unknown evidence index must not produce an anchor")
	}
}

func TestBoundedExcerptKeepsTablesWhole(t *testing.T) {
	table := "| a | b |\n| c | d |\n| e | f |\n| g | h |"
	if got := boundedExcerpt(table, "unrelated hint", 50); got != table {
		t.Errorf("table text must be returned whole, got %q", got)
	}
	// Prose is windowed around a hint token.
	prose := strings.Repeat("prefix filler ", 40) + "TARGET sentence here" + strings.Repeat(" trailing filler", 40)
	got := boundedExcerpt(prose, "TARGET", 120)
	if len(got) > 130 {
		t.Errorf("prose excerpt not bounded: %d chars", len(got))
	}
	if !strings.Contains(got, "TARGET") {
		t.Error("excerpt must contain the hint token")
	}
}

func TestSCABoostAdaptsVerdict(t *testing.T) {
	res := SCAResult{
		IsSufficient:   false,
		Confidence:     0.3,
		Contradictions: []string{"c1"},
		Claims: map[string]ClaimVerdict{
			"c1": {MissingInformation: []MissingPiece{{What: "the year"}}},
			"c2": {MissingInformation: []MissingPiece{{What: "the year"}}}, // duplicate
		},
	}
	b := res.Boost([]string{"fallback query"})
	if b.IsSufficient {
		t.Error("boost must carry is_sufficient=false")
	}
	if len(b.Missing) != 1 || b.Missing[0] != "the year" {
		t.Errorf("missing = %v, want the deduped 'the year'", b.Missing)
	}
	if !strings.Contains(b.Feedback, "the year") {
		t.Errorf("feedback = %q, must mention the missing piece", b.Feedback)
	}
	if len(b.Followups) != 1 || b.Followups[0] != "fallback query" {
		t.Errorf("followups = %v", b.Followups)
	}
}

// ---------------------------------------------------------------------------
// Query rewriter
// ---------------------------------------------------------------------------

func TestRewriteGapToQueryNeedsModelAndGaps(t *testing.T) {
	mdl := &stubJSONModel{value: map[string]any{"queries": []any{"q1"}}}
	if got := RewriteGapToQuery(context.Background(), RewriteDeps{Model: mdl}, "q", nil); got != nil {
		t.Error("no gaps must short-circuit")
	}
	if mdl.calls != 0 {
		t.Error("no gaps must not call the model")
	}
	if got := RewriteGapToQuery(context.Background(), RewriteDeps{}, "q",
		[]MissingPiece{{What: "x"}}); got != nil {
		t.Error("no model must return nil")
	}
}

func TestRewriteGapToQueryParsesAndDedupes(t *testing.T) {
	mdl := &stubJSONModel{value: map[string]any{"queries": []any{
		map[string]any{"query": "MASH finale run time minutes"},
		map[string]any{"question": "Cheers finale run time minutes"}, // key drift
		"plain string query",
		map[string]any{"query": "MASH finale run time minutes"}, // duplicate
		map[string]any{"query": "   "},                          // blank
	}}}
	got := RewriteGapToQuery(context.Background(), RewriteDeps{Model: mdl}, "which finale ran longest?",
		[]MissingPiece{{What: "runtimes", SearchHint: "finale run time"}})
	if len(got) != 3 {
		t.Fatalf("queries = %d, want 3 (blank + duplicate dropped)", len(got))
	}
	if got[0]["query"] != "MASH finale run time minutes" {
		t.Errorf("queries[0] = %q", got[0]["query"])
	}
	if got[1]["query"] != "Cheers finale run time minutes" {
		t.Errorf("queries[1] = %q, want the 'question' key accepted", got[1]["query"])
	}
	if got[2]["query"] != "plain string query" {
		t.Errorf("queries[2] = %q", got[2]["query"])
	}
}

func TestRewriteGapToQueryRendersAllVars(t *testing.T) {
	mdl := &stubJSONModel{value: map[string]any{"queries": []any{}}}
	RewriteGapToQuery(context.Background(), RewriteDeps{
		Model:           mdl,
		Prompts:         harness.StringPromptLoader{},
		BridgeValues:    []string{"M*A*S*H", "Cheers", "  "},
		ResearchContext: "tried: finale runtimes (nothing new)",
	}, "which finale ran longest?",
		[]MissingPiece{{What: "runtimes", SearchHint: "finale run time"}})

	// The fallback template uses the spaced Jinja form, and every var must be
	// substituted (an unreplaced {{ }} would corrupt the prompt).
	if strings.Contains(mdl.last, "{{") {
		t.Errorf("unsubstituted placeholder remains: %.200s", mdl.last)
	}
	for _, want := range []string{"which finale ran longest?", "- M*A*S*H", "- Cheers",
		"what: runtimes; hint: finale run time", "tried: finale runtimes"} {
		if !strings.Contains(mdl.last, want) {
			t.Errorf("prompt missing %q", want)
		}
	}
	// Blank bridge values are dropped.
	if strings.Contains(mdl.last, "-   ") {
		t.Error("blank bridge value must be dropped")
	}
}

func TestRewriteGapToQueryReturnsNilOnFailure(t *testing.T) {
	mdl := &stubJSONModel{err: errors.New("provider down")}
	if got := RewriteGapToQuery(context.Background(), RewriteDeps{Model: mdl}, "q",
		[]MissingPiece{{What: "x"}}); got != nil {
		t.Error("a model failure must return nil so the caller falls back")
	}
	// A non-dict response is also "no signal".
	mdl = &stubJSONModel{value: []any{"not", "a", "dict"}}
	if got := RewriteGapToQuery(context.Background(), RewriteDeps{Model: mdl}, "q",
		[]MissingPiece{{What: "x"}}); got != nil {
		t.Error("a non-dict response must return nil")
	}
}

// ---------------------------------------------------------------------------
// Prompt rendering
// ---------------------------------------------------------------------------

func TestRenderPromptUsesLoaderAndFallsBack(t *testing.T) {
	loader := harness.StringPromptLoader{"tpl": "hello {{ name }}!"}
	if got := renderPrompt(loader, "tpl", "fallback", map[string]string{"name": "world"}); got != "hello world!" {
		t.Errorf("renderPrompt = %q", got)
	}
	// Unknown template -> fallback.
	if got := renderPrompt(loader, "missing", "fallback text", nil); got != "fallback text" {
		t.Errorf("fallback = %q", got)
	}
	// Nil loader -> fallback.
	if got := renderPrompt(nil, "tpl", "fallback text", nil); got != "fallback text" {
		t.Errorf("nil loader = %q", got)
	}
	// Both {{k}} and {{ k }} are substituted (templates use the spaced form).
	if got := renderPrompt(harness.StringPromptLoader{"tpl": "a={{x}} b={{ y }}"}, "tpl", "",
		map[string]string{"x": "1", "y": "2"}); got != "a=1 b=2" {
		t.Errorf("spaced placeholder not substituted: %q", got)
	}
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func TestClampAndTruthy(t *testing.T) {
	if got := clamp(float64(1.5)); got != 1.0 {
		t.Errorf("clamp(1.5) = %v, want 1.0", got)
	}
	if got := clamp(float64(-0.2)); got != 0.0 {
		t.Errorf("clamp(-0.2) = %v, want 0.0", got)
	}
	if got := clamp("0.4"); got != 0.4 {
		t.Errorf("clamp(\"0.4\") = %v, want 0.4", got)
	}
	// Unparseable -> Python's default of 1.0.
	if got := clamp("not a number"); got != 1.0 {
		t.Errorf("clamp(bad) = %v, want 1.0", got)
	}
	for _, v := range []any{true, "true", "TRUE", "1", "yes", float64(1)} {
		if !truthy(v) {
			t.Errorf("truthy(%v) = false, want true", v)
		}
	}
	for _, v := range []any{false, "false", "0", "", float64(0), nil} {
		if truthy(v) {
			t.Errorf("truthy(%v) = true, want false", v)
		}
	}
}
