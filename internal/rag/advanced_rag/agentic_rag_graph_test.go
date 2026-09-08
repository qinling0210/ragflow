package advanced_rag

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/cloudwego/eino/schema"

	"ragflow/internal/rag/advanced_rag/harness"
)

// ---------------------------------------------------------------------------
// Test doubles
// ---------------------------------------------------------------------------

// scriptedModel answers with a fixed set of canned replies, one per call,
// looping on the last if exhausted. It satisfies harness.SessionModel. seen
// records every message slice handed to Complete, for assertions on retry turns.
type scriptedModel struct {
	mu      sync.Mutex
	replies []string
	idx     int
	seen    [][]schema.Message
}

func (m *scriptedModel) push(reply string) {
	m.replies = append(m.replies, reply)
}

func (m *scriptedModel) Complete(ctx context.Context, msgs []schema.Message, _ []harness.ToolSpec) (*harness.ModelReply, error) {
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.seen = append(m.seen, msgs)
	if len(m.replies) == 0 {
		return &harness.ModelReply{Content: ""}, nil
	}
	r := m.replies[m.idx]
	if m.idx < len(m.replies)-1 {
		m.idx++
	}
	return &harness.ModelReply{Content: r}, nil
}

// stubExecutor is a deterministic ToolExecutor recording every call.
type stubExecutor struct {
	mu        sync.Mutex
	calls     []string
	responses map[string]harness.ToolOutcome
}

func newStubExecutor() *stubExecutor {
	return &stubExecutor{responses: map[string]harness.ToolOutcome{}}
}

func (e *stubExecutor) add(name, payload string, status string) {
	e.responses[name] = harness.ToolOutcome{
		Status:  status,
		Payload: []any{payload},
	}
}

func (e *stubExecutor) Execute(ctx context.Context, name string, args map[string]any) (harness.ToolOutcome, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.calls = append(e.calls, name)
	if o, ok := e.responses[name]; ok {
		return o, nil
	}
	return harness.ToolOutcome{Status: harness.StatusOK, Payload: []any{"{}"}}, nil
}

func newToolset(exec *stubExecutor) *harness.Toolset {
	return &harness.Toolset{
		ThinkingMode:  "high",
		HasWebSearch:  true,
		DisabledTools: map[string]bool{},
		Exec:          exec,
	}
}

// countingRetriever is a no-op retriever recording the requested top-N. It
// satisfies harness.Retriever (= tools.Retriever).
type countingRetriever struct {
	mu   sync.Mutex
	topN int
}

func (r *countingRetriever) Retrieve(_ context.Context, req harness.RetrieveRequest) ([]map[string]any, error) {
	r.mu.Lock()
	r.topN = req.TopN
	r.mu.Unlock()
	return []map[string]any{{"doc_id": "d1", "docnm_kwd": "doc1", "content": "Saint Lawrence River; 14 April 1865; 9 December 2019."}}, nil
}

// ---------------------------------------------------------------------------
// Unit tests for the routing / helpers (no init() required).
// ---------------------------------------------------------------------------

func TestRouteSCAAlwaysCloseoutOnNoProgress(t *testing.T) {
	// NoProgress is the hard stop: regardless of SCA enablement or verdict, the
	// loop must close out rather than spin another round.
	st := &AgenticState{Verdict: VerdictInsufficient, MaxLoops: 3, NoProgress: true}
	if n := routeSCA(st, true, 3); n != nodeFormalizeAnswer {
		t.Fatalf("SCA on: expected formalize on no-progress, got %v", n)
	}
	if n := routeSCA(st, false, 3); n != nodeFormalizeAnswer {
		t.Fatalf("SCA off: expected formalize on no-progress, got %v", n)
	}
}

func TestRouteSCADisabledSCAClosesOut(t *testing.T) {
	// With SCA disabled the single research pass' verdict is informational; we
	// never start a rewrite round.
	st := &AgenticState{Verdict: VerdictInsufficient, MaxLoops: 3}
	if n := routeSCA(st, false, 3); n != nodeFormalizeAnswer {
		t.Fatalf("expected formalize when SCA disabled, got %v", n)
	}
}

func TestRouteSCAInsufficientStartsRewrite(t *testing.T) {
	// Sufficient verdict closes out.
	st := &AgenticState{Verdict: VerdictSufficient, MaxLoops: 3}
	if n := routeSCA(st, true, 3); n != nodeFormalizeAnswer {
		t.Fatalf("expected formalize on sufficient, got %v", n)
	}
	// Insufficient + rounds under cap + headroom remaining => rewrite.
	st = &AgenticState{Verdict: VerdictInsufficient, MaxLoops: 3, SearchRounds: 0}
	st.Deadline = time.Now().Add(120 * time.Second) // 120s remaining >> MinRoundHeadroomS
	if n := routeSCA(st, true, 3); n != nodeQueryRewrite {
		t.Fatalf("expected rewrite on insufficient+headroom, got %v", n)
	}
	// At max rounds => formalize.
	st = &AgenticState{Verdict: VerdictInsufficient, MaxLoops: 3, SearchRounds: 3}
	st.Deadline = time.Now().Add(120 * time.Second)
	if n := routeSCA(st, true, 3); n != nodeFormalizeAnswer {
		t.Fatalf("expected formalize at max rounds, got %v", n)
	}
}

func TestSelectSCAViewIdentity(t *testing.T) {
	chunks := make([]map[string]any, 0, 80)
	for i := 0; i < 80; i++ {
		chunks = append(chunks, map[string]any{
			"chunk_id":   fmt.Sprintf("c%d", i),
			"content":    fmt.Sprintf("the quick brown fox %d", i),
			"similarity": 1.0 - float64(i)/100.0,
		})
	}
	v1, id1 := SelectSCAView(chunks, []string{"quick", "brown"})
	if len(v1) != SCAViewCap {
		t.Fatalf("expected view cap %d, got %d", SCAViewCap, len(v1))
	}
	// Same inputs => same identity (no Process-randomised hash).
	_, id2 := SelectSCAView(chunks, []string{"quick", "brown"})
	if id1 != id2 {
		t.Fatalf("view identity should be deterministic, got %q vs %q", id1, id2)
	}
	// An enlarged pool with the SAME selected ids keeps the same identity — this
	// is exactly the "review view unchanged" early-exit signal.
	more := append(chunks, map[string]any{"chunk_id": "extra", "content": "unrelated tail", "similarity": 0.0})
	_, id3 := SelectSCAView(more, []string{"quick", "brown"})
	if id1 != id3 {
		t.Fatalf("adding non-selected chunks should NOT change the view identity, got %q vs %q", id1, id3)
	}
}

func TestMergeSlotPatchAdoptsStronger(t *testing.T) {
	strong := 0.95
	base := harness.NewState([]harness.Variable{
		{ID: 0, Type: "aspect", Candidate: strPtr("base"), CandidateStrength: &strong},
	}, 0, nil)
	// A weak branch (0.30) must NOT downgrade the strong base.
	weak := 0.30
	branch := harness.NewState([]harness.Variable{
		{ID: 0, Type: "aspect", Candidate: strPtr("weak-candidate"), CandidateStrength: &weak},
	}, 1, nil)
	merged := MergeSlotPatch(base, branch)
	// A weak branch does not downgrade the strong base, AND produces no change
	// at all — MergeSlotPatch returns nil to signal "no new patch".
	if merged != nil {
		t.Fatalf("weak branch over strong base should yield NO change (nil patch), got %+v", merged)
	}

	// A stronger branch wins.
	stronger := 0.99
	branch2 := harness.NewState([]harness.Variable{
		{ID: 0, Type: "aspect", Candidate: strPtr("strong-candidate"), CandidateStrength: &stronger},
	}, 1, nil)
	merged2 := MergeSlotPatch(base, branch2)
	if merged2 == nil {
		t.Fatal("expected a merge result")
	}
	if got := merged2.State[0].Candidate; got == nil || *got != "strong-candidate" {
		t.Fatalf("strong branch should upgrade, got %v", got)
	}
}

func TestMergeSlotPatchNoChangeReturnsNil(t *testing.T) {
	base := harness.NewState([]harness.Variable{
		{ID: 0, Type: "aspect", Candidate: strPtr("same")},
	}, 0, nil)
	branch := harness.NewState([]harness.Variable{
		{ID: 0, Type: "aspect", Candidate: strPtr("same")},
	}, 1, nil)
	if m := MergeSlotPatch(base, branch); m != nil {
		t.Fatalf("identical patch should return nil, got %+v", m)
	}
}

func TestExpandFanoutsParsesJSON(t *testing.T) {
	mdl := &scriptedModel{}
	mdl.push(`{"fanouts": ["who discovered it", "when was it discovered"]}`)
	got := ExpandFanouts(context.Background(), RAGTools{Model: mdl}, "What was discovered and by whom?")
	if len(got) != 2 {
		t.Fatalf("expected 2 fan-outs, got %v", got)
	}
	if got[0] != "who discovered it" || got[1] != "when was it discovered" {
		t.Fatalf("unexpected fan-outs: %v", got)
	}
}

func TestExpandFanoutsFallbackOnBadJSON(t *testing.T) {
	mdl := &scriptedModel{}
	mdl.push("I cannot break this down.")
	got := ExpandFanouts(context.Background(), RAGTools{Model: mdl}, "single question")
	// Bad JSON: the loop falls back to line-splitting the model's reply, which
	// yields the reply itself (single line), not the raw question.
	if len(got) != 1 || got[0] != "I cannot break this down." {
		t.Fatalf("expected line-split fallback, got %v", got)
	}
}

func TestExpandFanoutsNilModelReturnsQuestion(t *testing.T) {
	got := ExpandFanouts(context.Background(), RAGTools{}, "single question")
	if len(got) != 1 || got[0] != "single question" {
		t.Fatalf("expected raw question when no model, got %v", got)
	}
}

func TestRenderSlotDraftShowsUnresolved(t *testing.T) {
	strong := 0.9
	st := harness.NewState([]harness.Variable{
		{ID: 0, Type: "aspect", Candidate: strPtr("answer A"), CandidateStrength: &strong},
		{ID: 1, Type: "aspect"}, // empty
	}, 0, nil)
	draft := RenderSlotDraft(st, "", map[string]SlotEvidence{})
	if !strings.Contains(draft, "answer A") {
		t.Fatalf("draft missing filled candidate: %q", draft)
	}
	if !strings.Contains(draft, "UNRESOLVED") {
		t.Fatalf("draft missing unresolved marker: %q", draft)
	}
}

func TestFanoutSearchInvokesExecutor(t *testing.T) {
	ctx := context.Background()
	exec := newStubExecutor()
	deps := RAGTools{Tools: newToolset(exec)}
	st := &AgenticState{KB: &harness.Kbinfos{}}
	added := FanoutSearch(ctx, deps, st, []string{"q1", "q2"}, 8, 60)
	// The stub executor does not merge results back into st.KB, so the delta is
	// 0; what we assert is that every query reached the executor exactly once.
	if added != 0 {
		t.Fatalf("stub executor leaves KB unchanged; expected 0 delta, got %d", added)
	}
	if len(exec.calls) != 2 {
		t.Fatalf("expected 2 executor calls, got %d", len(exec.calls))
	}
}

func TestSCAGapsToRewriteFromSubQueries(t *testing.T) {
	sca := map[string]any{
		"sub_queries": []any{
			map[string]any{"sub_query": "q1", "satisfied": true, "missing_fact": "", "search_hint": ""},
			map[string]any{"sub_query": "q2", "satisfied": false, "missing_fact": "the year", "search_hint": "search for dates"},
		},
	}
	gaps := SCAGapsToRewrite(sca)
	if len(gaps) != 1 {
		t.Fatalf("expected 1 gap, got %d: %+v", len(gaps), gaps)
	}
	if gaps[0].What != "the year" || gaps[0].SearchHint != "search for dates" {
		t.Fatalf("unexpected gap: %+v", gaps[0])
	}
}

func TestSCAGapsToRewriteFallbackToClaims(t *testing.T) {
	sca := map[string]any{
		"claims": map[string]any{
			"c1": map[string]any{
				"verdict": "unverified",
				"missing_information": []any{
					map[string]any{"what": "population", "search_hint": "demographics"},
				},
			},
		},
	}
	gaps := SCAGapsToRewrite(sca)
	if len(gaps) != 1 {
		t.Fatalf("expected 1 gap from claims fallback, got %d: %+v", len(gaps), gaps)
	}
	if gaps[0].What != "population" {
		t.Fatalf("unexpected gap: %+v", gaps[0])
	}
}

func TestRunReturnsNonNilState(t *testing.T) {
	ctx := context.Background()
	mdl := &scriptedModel{}
	mdl.push(`{"fanouts": ["when built", "where located"]}`)
	mdl.push(`{"slots":[{"id":0,"type":"aspect","question":"when built","clues":["1865"]},{"id":1,"type":"aspect","question":"where located","clues":["geneva"]}], "first_queries":["when built","where located"]}`)
	mdl.push("Built in 1865.")
	mdl.push("In Geneva.")
	mdl.push(`{"is_sufficient": true, "score": 0.9, "contradictions": [], "reasoning": "ok", "claims": {}}`)

	exec := newStubExecutor()
	exec.add("search_chunks", `{"hit":[{"doc_id":"d1","docnm_kwd":"doc1","content":"Built 1865, Geneva."}],"doc_aggs":[]}`, harness.StatusOK)

	st := BuildAgenticGraph(ctx, RAGTools{
		Model:  mdl,
		Tools:  newToolset(exec),
		Logger: log.Default(),
	}, "When and where was it built?", "", 3, nil)

	if st == nil {
		t.Fatal("expected a non-nil AgenticState")
	}
	if strings.TrimSpace(st.Draft) == "" {
		t.Fatal("expected a non-empty research draft")
	}
	if st.NoProgress {
		t.Fatal("a completed run should not report NoProgress")
	}
}

// ---------------------------------------------------------------------------
// Explicit-wiring integration tests (replaces the old init()-based registration
// checks). Activation is explicit, mirroring Python's dialog_service.py
// instantiating RAGTools: the test registers the loop before calling Run.
// ---------------------------------------------------------------------------

// TestMain keeps the package's tests on in-memory doubles: graph exploration
// otherwise seeds its dense search from the tenant embedding model, which
// reaches a database these tests do not have. Returning nil keeps the keyword
// fallback, which is the path under test.
func TestMain(m *testing.M) {
	harness.SetSeedEncoder(func(context.Context, string, string) []float64 { return nil })
	os.Exit(m.Run())
}

func TestNewAgenticLoopDrivesHarnessRun(t *testing.T) {
	SetAgenticLoop(NewAgenticLoop())
	defer SetAgenticLoop(nil)

	facts := `{"fact":"Saint Lawrence River; 14 April 1865; 9 December 2019."}`
	if err := json.Unmarshal([]byte(facts), &struct{}{}); err != nil {
		t.Fatalf("bad fixture: %v", err)
	}

	mdl := &scriptedModel{}
	mdl.push(`{"fanouts": ["when was it opened", "what river"]}`)
	mdl.push(`{"slots":[{"id":0,"type":"aspect","question":"when opened","clues":["1865"]},{"id":1,"type":"aspect","question":"which river","clues":["river"]}], "first_queries":["when opened","which river"]}`)
	mdl.push("It opened on 14 April 1865.")
	mdl.push("The Saint Lawrence River.")
	mdl.push(`{"is_sufficient": true, "score": 0.9, "contradictions": [], "reasoning": "ok", "claims": {}}`)

	deps := RAGTools{
		Retriever: &countingRetriever{},
		Model:     mdl,
		Prompts:   &fakePrompts{},
	}

	got := runHarnessRun(t, deps, "When did it open and on which river?")
	if got == nil {
		t.Fatal("expected a non-nil RunResponse")
	}
	if got.Mode.Label != "high" {
		t.Fatalf("expected high mode, got %q", got.Mode.Label)
	}
	// The agentic loop routed through the planner + slot research; at minimum the
	// response is produced without error.
	if len(got.Chunks) == 0 {
		t.Fatalf("expected >=1 chunk (retriever stub fills KB), got %d", len(got.Chunks))
	}
}

func TestNewAgenticLoopDegradesWithoutModel(t *testing.T) {
	SetAgenticLoop(NewAgenticLoop())
	defer SetAgenticLoop(nil)

	deps := RAGTools{
		Retriever: &countingRetriever{},
		Model:     nil, // no model => direct fallback
		Prompts:   &fakePrompts{},
	}

	got := runHarnessRun(t, deps, "Any question?")
	if got == nil {
		t.Fatal("expected a non-nil RunResponse")
	}
	// With no model the loop falls back to a single direct retrieval; no slots.
	if len(got.Slots) != 0 {
		t.Fatalf("expected 0 slots when no model, got %d", len(got.Slots))
	}
	if len(got.Chunks) == 0 {
		t.Fatalf("expected >=1 chunk from direct fallback, got %d", len(got.Chunks))
	}
}

func TestNewAgenticLoopRoundsToggleDeactivates(t *testing.T) {
	// After deregistering (nil), Run must NOT enter the agentic loop.
	SetAgenticLoop(nil)

	mdl := &scriptedModel{}
	mdl.push(`{"fanouts": ["x", "y"]}`) // would only be produced by the loop's planner
	deps := RAGTools{
		Retriever: &countingRetriever{},
		Model:     mdl,
		Prompts:   &fakePrompts{},
	}
	got := runHarnessRun(t, deps, "Question?")
	if got == nil {
		t.Fatal("expected a non-nil RunResponse")
	}
	// In single-session fallback the planner fan-out prompt is never issued, so
	// the scripted model's first reply is never consumed.
	if mdl.idx != 0 {
		t.Fatalf("expected agentic planner NOT to run (model idx %d, want 0)", mdl.idx)
	}
	_ = got
}

// runHarnessRun drives Run in high mode and returns the response. It is
// the Go analogue of dialog_service.py invoking RAGTools for an agentic answer.
func runHarnessRun(t *testing.T, deps RAGTools, question string) *RunResponse {
	t.Helper()
	req := harness.RunRequest{
		Question:     question,
		DatasetIDs:   []string{"kb1"},
		ThinkingMode: "high",
	}
	resp := Rag(context.Background(), deps, req)
	if resp == nil {
		t.Fatalf("Run returned nil response")
	}
	return resp
}

// strPtr / fltPtr / fakePrompts are tiny helpers shared across the tests.
func strPtr(s string) *string { return &s }

type fakePrompts struct{}

func (f *fakePrompts) Load(name string) (string, error) { return "", nil }

// Test doubles for the RAGTools method tests. Kept local to the advanced_rag package so
// the tests can call Run/ComposeAnswer (which live here) without importing agent
// from the harness test package (that would be an import cycle).

// fakeModel replays a scripted sequence of replies, so the session's control
// flow can be driven without a provider.
type fakeModel struct {
	replies  []*harness.ModelReply
	calls    int
	messages []schema.Message
}

func (f *fakeModel) Complete(_ context.Context, msgs []schema.Message, _ []harness.ToolSpec) (*harness.ModelReply, error) {
	f.messages = msgs
	if f.calls >= len(f.replies) {
		return &harness.ModelReply{Content: `<state>{"new_states": []}</state>`}, nil
	}
	r := f.replies[f.calls]
	f.calls++
	return r, nil
}

// lastUserPrompt returns the content of the most recent user message, so tests
// can assert on the prompt the model actually received.
func (f *fakeModel) lastUserPrompt() string {
	for i := len(f.messages) - 1; i >= 0; i-- {
		if f.messages[i].Role == schema.User {
			return f.messages[i].Content
		}
	}
	return ""
}

// corpusRetriever answers every query from a fixed corpus.
type corpusRetriever struct{ calls []string }

func (c *corpusRetriever) Retrieve(_ context.Context, req harness.RetrieveRequest) ([]map[string]any, error) {
	c.calls = append(c.calls, req.Query)
	return []map[string]any{{
		"chunk_id":   "c1",
		"content":    "Culdcept was created by OmiyaSoft and released in 1999.",
		"doc_id":     "doc-culdcept",
		"doc_name":   "Culdcept History",
		"dataset_id": "kb1",
	}}, nil
}

// emptyRetriever simulates a corpus with no matches.
type emptyRetriever struct{}

func (emptyRetriever) Retrieve(_ context.Context, _ harness.RetrieveRequest) ([]map[string]any, error) {
	return nil, nil
}

func TestRunLowModeIsSinglePass(t *testing.T) {
	r := &corpusRetriever{}
	resp := Rag(context.Background(), RAGTools{Retriever: r}, harness.RunRequest{
		Question:     "Who created Culdcept?",
		ThinkingMode: "low",
		DatasetIDs:   []string{"kb1"},
		UseCompiled:  true,
	})
	if resp.EmptyResult {
		t.Fatal("low mode must retrieve evidence")
	}
	if len(resp.Chunks) != 1 {
		t.Fatalf("chunks = %d, want 1", len(resp.Chunks))
	}
	if len(r.calls) != 1 {
		t.Errorf("retrieval calls = %d, want 1 (low is a single pass)", len(r.calls))
	}
	if resp.Mode.Agentic {
		t.Error("low must resolve to a non-agentic spec")
	}
	if len(resp.DocAggs) != 1 || resp.DocAggs[0]["doc_id"] != "doc-culdcept" {
		t.Fatalf("doc_aggs = %v", resp.DocAggs)
	}
	if resp.DocAggs[0]["doc_name"] != "Culdcept History" {
		t.Errorf("doc_name = %v, want 'Culdcept History'", resp.DocAggs[0]["doc_name"])
	}
	if harness.MemorySize(resp.Kbinfos) != 1 {
		t.Errorf("memory size = %d, want 1", harness.MemorySize(resp.Kbinfos))
	}
}

func TestRunUnknownModeFallsBackToNaive(t *testing.T) {
	r := &corpusRetriever{}
	resp := Rag(context.Background(), RAGTools{Retriever: r}, harness.RunRequest{
		Question:     "Who created Culdcept?",
		ThinkingMode: "not-a-real-mode",
		DatasetIDs:   []string{"kb1"},
	})
	if resp.Mode.Label != "naive" {
		t.Errorf("mode = %q, want naive", resp.Mode.Label)
	}
	if resp.EmptyResult {
		t.Error("naive must still retrieve (it degrades, not fails)")
	}
}

func TestRunAgenticDegradesToDirectWithoutModel(t *testing.T) {
	r := &corpusRetriever{}
	resp := Rag(context.Background(), RAGTools{Retriever: r}, harness.RunRequest{
		Question:     "Who created Culdcept?",
		ThinkingMode: "high",
		DatasetIDs:   []string{"kb1"},
	})
	if resp.EmptyResult {
		t.Error("agentic mode without a model must degrade to direct search")
	}
}

func TestRunAgenticSeedsSlotsAndRunsSession(t *testing.T) {
	r := &corpusRetriever{}
	mdl := &fakeModel{replies: []*harness.ModelReply{
		{Content: `{"slots": [{"id": 0, "type": "entity", "clues": ["creator"]}, {"id": 1, "type": "date", "clues": ["released"]}], "first_queries": ["creator"]}`},
		{Content: "", ToolCalls: []harness.ToolCall{{ID: "c0", Name: "retrieve", Args: map[string]any{"query": []any{"Culdcept creator"}}}}},
		{Content: `<state>{"new_states":[{"state":[{"id":0,"candidate":"OmiyaSoft","candidate_strength":0.95}]}]}</state>`},
	}}
	resp := Rag(context.Background(), RAGTools{Retriever: r, Model: mdl}, harness.RunRequest{
		Question:     "Who created Culdcept and when?",
		ThinkingMode: "medium",
		DatasetIDs:   []string{"kb1"},
	})
	if resp.EmptyResult {
		t.Fatal("the session's retrieval must land in kbinfos")
	}
	if len(resp.Slots) != 2 {
		t.Fatalf("slots = %d, want 2 (from the seeded table)", len(resp.Slots))
	}
	filled := 0
	for _, v := range resp.Slots {
		if v.Filled() {
			filled++
		}
	}
	if filled != 1 {
		t.Errorf("filled slots = %d, want 1 (the patched candidate)", filled)
	}
	if len(r.calls) == 0 {
		t.Error("the retrieve tool call never reached the retriever")
	}
}

func TestRunNeverPanicsWithoutBackend(t *testing.T) {
	resp := Rag(context.Background(), RAGTools{}, harness.RunRequest{
		Question:     "anything",
		ThinkingMode: "low",
		DatasetIDs:   []string{"kb1"},
	})
	if !resp.EmptyResult {
		t.Error("an unavailable backend must yield an empty result, not an error")
	}
}

func TestRunLowReturnsComposedAnswer(t *testing.T) {
	mdl := &fakeModel{replies: []*harness.ModelReply{
		{Content: `{"entity": ["Culdcept"], "aliases": [], "fact_type": [], "qualifiers": []}`},
		{Content: "Culdcept was created by OmiyaSoft and released in 1999 [1]."},
	}}
	resp := Rag(context.Background(), RAGTools{
		Retriever: &corpusRetriever{},
		Model:     mdl,
	}, harness.RunRequest{
		Question:     "Who created Culdcept?",
		ThinkingMode: "low",
		DatasetIDs:   []string{"kb1"},
	})
	if resp.EmptyResult {
		t.Fatal("low must retrieve evidence")
	}
	if resp.Answer == "" {
		t.Fatal("low must return a composed answer (was empty before the composition node)")
	}
	if !strings.Contains(resp.Answer, "OmiyaSoft") {
		t.Errorf("answer = %q, want it grounded in the retrieved evidence", resp.Answer)
	}
	if len(resp.Chunks) == 0 || len(resp.DocAggs) == 0 {
		t.Error("chunks/aggs must be reported alongside the answer")
	}
}

func TestRunUnknownModeReturnsComposedAnswer(t *testing.T) {
	mdl := &fakeModel{replies: []*harness.ModelReply{{Content: "OmiyaSoft [1]"}}}
	resp := Rag(context.Background(), RAGTools{
		Retriever: &corpusRetriever{},
		Model:     mdl,
	}, harness.RunRequest{
		Question:     "Who created Culdcept?",
		ThinkingMode: "not-a-mode",
		DatasetIDs:   []string{"kb1"},
	})
	if resp.Mode.Label != "naive" {
		t.Fatalf("mode = %q, want naive", resp.Mode.Label)
	}
	if resp.Answer == "" {
		t.Error("naive must compose an answer")
	}
}

func TestRunEmptyResponseShortCircuits(t *testing.T) {
	mdl := &fakeModel{}
	resp := Rag(context.Background(), RAGTools{
		Retriever:     &emptyRetriever{},
		Model:         mdl,
		EmptyResponse: "I don't have enough information.",
	}, harness.RunRequest{
		Question:     "Who created Culdcept?",
		ThinkingMode: "low",
		DatasetIDs:   []string{"kb1"},
	})
	if !resp.EmptyResult {
		t.Fatal("expected no evidence")
	}
	if resp.Answer != "I don't have enough information." {
		t.Errorf("answer = %q, want the configured empty response", resp.Answer)
	}
	if mdl.calls != 0 {
		t.Errorf("model calls = %d, want 0 (short-circuited before composition)", mdl.calls)
	}
}

func TestRunCanDisableComposition(t *testing.T) {
	no := false
	resp := Rag(context.Background(), RAGTools{
		Retriever:     &corpusRetriever{},
		Model:         &fakeModel{},
		ComposeAnswer: &no,
	}, harness.RunRequest{
		Question:     "Who created Culdcept?",
		ThinkingMode: "low",
		DatasetIDs:   []string{"kb1"},
	})
	if resp.Answer != "" {
		t.Errorf("answer = %q, want empty when composition is disabled", resp.Answer)
	}
	if len(resp.Chunks) == 0 {
		t.Error("evidence must still be returned")
	}
}

func TestRunFormalizesFromConversation(t *testing.T) {
	mdl := &fakeModel{replies: []*harness.ModelReply{
		{Content: `{"question": "When was Culdcept released?", "keywords": "Culdcept, release, released"}`},
		{Content: "Culdcept was released in 1999 [1]."},
	}}
	retriever := &corpusRetriever{}
	resp := Rag(context.Background(), RAGTools{
		Retriever: retriever,
		Model:     mdl,
		Messages: []schema.Message{
			*schema.UserMessage("Who created Culdcept?"),
			*schema.AssistantMessage("OmiyaSoft.", nil),
			*schema.UserMessage("When was it released?"),
		},
	}, harness.RunRequest{
		Question:     "When was it released?",
		ThinkingMode: "low",
		DatasetIDs:   []string{"kb1"},
	})
	if len(retriever.calls) == 0 {
		t.Fatal("retriever was never called")
	}
	if !strings.HasPrefix(retriever.calls[0], "When was Culdcept released?") {
		t.Errorf("retrieval query = %q, want it to start with the formalized question", retriever.calls[0])
	}
	if !strings.Contains(retriever.calls[0], "Culdcept") {
		t.Errorf("retrieval query = %q, want the formalized keywords appended", retriever.calls[0])
	}
	if resp.Answer == "" {
		t.Error("answer must be composed after formalization")
	}
}

func TestRunFormalizeDisabledByDefault(t *testing.T) {
	mdl := &fakeModel{replies: []*harness.ModelReply{{Content: "answer [1]"}}}
	Rag(context.Background(), RAGTools{
		Retriever: &corpusRetriever{},
		Model:     mdl,
		Messages: []schema.Message{
			*schema.UserMessage("Who created Culdcept?"),
			*schema.AssistantMessage("OmiyaSoft.", nil),
			*schema.UserMessage("When was it released?"),
		},
	}, harness.RunRequest{
		Question:     "When was it released?",
		ThinkingMode: "low",
		DatasetIDs:   []string{"kb1"},
	})
	if mdl.calls != 1 {
		t.Errorf("model calls = %d, want 1 (formalize is opt-in)", mdl.calls)
	}
}

func TestRunAgenticComposesFromResearchFindings(t *testing.T) {
	kb := &harness.Kbinfos{
		Chunks:     []map[string]any{{"chunk_id": "c1", "content": "Culdcept: OmiyaSoft, 1999.", "similarity": 0.9}},
		PreSummary: "Culdcept was created by OmiyaSoft and released in 1999.",
	}
	mdl := &fakeModel{replies: []*harness.ModelReply{{Content: "Culdcept was created by OmiyaSoft and released in 1999 [1]."}}}
	res := ComposeAnswer(context.Background(), AnswerDeps{Model: mdl}, kb, "Who created Culdcept?", false, false)
	if res.Failed {
		t.Fatal("composition must succeed")
	}
	if !strings.Contains(mdl.lastUserPrompt(), "Culdcept was created by OmiyaSoft and released in 1999.") {
		t.Errorf("prompt missing the research findings:\n%s", mdl.lastUserPrompt())
	}
}

// userTurnAt returns the Content of the last message of the idx-th Complete
// call (0-based), i.e. the user turn of that attempt.
func userTurnAt(m *scriptedModel, idx int) string {
	m.mu.Lock()
	defer m.mu.Unlock()
	if idx >= len(m.seen) || len(m.seen[idx]) == 0 {
		return ""
	}
	return m.seen[idx][len(m.seen[idx])-1].Content
}

// TestGenJSONUsesOutputNewlineUserTurn locks the contract that every JSONModel
// round mirrors Python gen_json's "Output:\n" user turn. It was previously an
// invisible convention buried inside jsonModelAdapter.GenJSON — a second JSONModel
// implementation could silently drop it and diverge from Python without any test
// catching it. See query_rewriter.go parity note (diff ②).
func TestGenJSONUsesOutputNewlineUserTurn(t *testing.T) {
	mdl := &scriptedModel{}
	mdl.push("{\"ok\": true}")

	if _, err := (&jsonModelAdapter{inner: mdl}).GenJSON(context.Background(), "Render JSON."); err != nil {
		t.Fatalf("GenJSON failed: %v", err)
	}
	if got := userTurnAt(mdl, 0); got != "Output:\n" {
		t.Errorf("first-round user turn = %q, want exactly %q (Python gen_json separator)", got, "Output:\n")
	}
}

func TestGenJSONRetriesMalformedReplyWithCorrection(t *testing.T) {
	mdl := &scriptedModel{}
	// First round: fenced but unparseable JSON. Second round: well-formed.
	mdl.push("```json\n{\"queries\": [broken")
	mdl.push("{\"queries\": [{\"query\": \"q1\"}]}")

	got, err := (&jsonModelAdapter{inner: mdl}).GenJSON(context.Background(), "Rewrite the query.")
	if err != nil {
		t.Fatalf("GenJSON failed after corrective retry: %v", err)
	}
	m, ok := got.(map[string]any)
	if !ok {
		t.Fatalf("GenJSON returned %T, want map[string]any", got)
	}
	queries, ok := m["queries"].([]any)
	if !ok || len(queries) != 1 {
		t.Fatalf("queries = %#v, want a 1-element list", m["queries"])
	}

	// The retry must feed the previous answer and the parse error back to the
	// model, mirroring gen_json's corrective prompt.
	turn := userTurnAt(mdl, 1)
	for _, want := range []string{"Generated JSON is as following:", "broken", "But exception while loading:", "Please reconsider and correct it."} {
		if !strings.Contains(turn, want) {
			t.Errorf("retry user turn missing %q; got: %s", want, turn)
		}
	}
}

func TestGenJSONStripsThinkAndFenceOnFirstTry(t *testing.T) {
	mdl := &scriptedModel{}
	mdl.push("Sure.\n<thinking>drafting...</thinking>\n```json\n{\"ok\": true}\n```\n")

	got, err := (&jsonModelAdapter{inner: mdl}).GenJSON(context.Background(), "Render JSON.")
	if err != nil {
		t.Fatalf("GenJSON failed on a wrapper-padded reply: %v", err)
	}
	m, ok := got.(map[string]any)
	if !ok {
		t.Fatalf("GenJSON returned %T, want map[string]any", got)
	}
	if m["ok"] != true {
		t.Errorf(`m["ok"] = %#v, want true`, m["ok"])
	}
	if len(mdl.seen) != 1 {
		t.Errorf("expected a single attempt, got %d", len(mdl.seen))
	}
}

func TestGenJSONGivesUpAfterMaxRetries(t *testing.T) {
	mdl := &scriptedModel{}
	mdl.push("I cannot produce JSON today.")
	mdl.push("Still not JSON.")

	_, err := (&jsonModelAdapter{inner: mdl}).GenJSON(context.Background(), "Rewrite the query.")
	if err == nil {
		t.Fatal("GenJSON succeeded, want error after exhausting retries")
	}
	if len(mdl.seen) != 2 {
		t.Errorf("expected 2 attempts (genJSONMaxRetry), got %d", len(mdl.seen))
	}
}

// errOnceModel fails the underlying chat call and counts invocations, proving
// chat/transport errors are NOT retried (gen_json propagates them immediately).
type errOnceModel struct {
	n int
}

func (m *errOnceModel) Complete(_ context.Context, _ []schema.Message, _ []harness.ToolSpec) (*harness.ModelReply, error) {
	m.n++
	return nil, errors.New("provider down")
}

func TestGenJSONPropagatesChatErrorWithoutRetry(t *testing.T) {
	mdl := &errOnceModel{}
	_, err := (&jsonModelAdapter{inner: mdl}).GenJSON(context.Background(), "Rewrite the query.")
	if err == nil {
		t.Fatal("GenJSON succeeded, want the chat error")
	}
	if mdl.n != 1 {
		t.Errorf("chat invoked %d times, want exactly 1 (no retry on transport error)", mdl.n)
	}
}

func TestGenJSONNilModel(t *testing.T) {
	if _, err := (&jsonModelAdapter{inner: nil}).GenJSON(context.Background(), "x"); err == nil {
		t.Fatal("GenJSON with a nil inner model succeeded, want error")
	}
}
