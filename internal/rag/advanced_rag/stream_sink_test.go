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

package advanced_rag

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/cloudwego/eino/schema"

	"ragflow/internal/rag/advanced_rag/harness"
)

// streamingModel is a SessionModel that can also stream.
type streamingModel struct {
	pieces []string
	fail   bool
}

func (m *streamingModel) Complete(_ context.Context, _ []schema.Message, _ []harness.ToolSpec) (*harness.ModelReply, error) {
	if m.fail {
		return nil, errors.New("boom")
	}
	out := ""
	for _, p := range m.pieces {
		out += p
	}
	return &harness.ModelReply{Content: out}, nil
}

func (m *streamingModel) StreamComplete(_ context.Context, _ []schema.Message, _ []harness.ToolSpec, onDelta func(string, bool) error) (*harness.ModelReply, error) {
	if m.fail {
		return nil, errors.New("stream boom")
	}
	out := ""
	for _, p := range m.pieces {
		if onDelta != nil {
			if err := onDelta(p, false); err != nil {
				return nil, err
			}
		}
		out += p
	}
	return &harness.ModelReply{Content: out}, nil
}

func TestComposeAnswerStreamForwardsDeltas(t *testing.T) {
	m := &streamingModel{pieces: []string{"Hello ", "world"}}
	var got []string
	res, err := ComposeAnswerStream(context.Background(), AnswerDeps{Model: m}, m, nil, "q", false,
		func(delta string, _ bool) error {
			got = append(got, delta)
			return nil
		})

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.Answer != "Hello world" {
		t.Fatalf("Answer = %q, want %q", res.Answer, "Hello world")
	}
	if len(got) != 2 || got[0] != "Hello " || got[1] != "world" {
		t.Fatalf("deltas = %v, want the two pieces in order", got)
	}
}

func TestComposeAnswerStreamReturnsErrorOnFailure(t *testing.T) {
	m := &streamingModel{fail: true}

	if _, err := ComposeAnswerStream(context.Background(), AnswerDeps{Model: m}, m, nil, "q", false, nil); err == nil {
		t.Fatal("want an error so the caller can fall back to one-shot")
	}
}

func TestAnswerSinkDeliverAndReset(t *testing.T) {
	var got []string
	resets := 0
	s := &AnswerSink{
		OnDelta: func(delta string, _ bool) { got = append(got, delta) },
		OnReset: func() { resets++; got = nil },
	}

	s.deliver("a", false)
	s.deliver("b", false)
	s.reset()

	if len(got) != 0 || resets != 1 {
		t.Fatalf("after reset: got=%v resets=%d, want empty and 1", got, resets)
	}
	s.deliver("", false) // empty deltas are dropped
	if len(got) != 0 {
		t.Fatalf("got = %v, want empty delta dropped", got)
	}
}

func TestNilAnswerSinkIsInert(t *testing.T) {
	var s *AnswerSink

	s.deliver("x", false) // must not panic
	s.reset()
}

func TestSessionCacheIsSharedPerSession(t *testing.T) {
	a := SessionCache("session-A")
	b := SessionCache("session-A")
	c := SessionCache("session-B")

	if a != b {
		t.Fatal("the same session must reuse one cache")
	}
	if a == c {
		t.Fatal("different sessions must not share a cache")
	}
	a.Store("population of Paris 2019", "2.1 million")
	if _, ok := b.Lookup("legal population of Paris in 2019"); !ok {
		t.Fatal("a re-ask in the same session must hit the cache")
	}
	if _, ok := c.Lookup("legal population of Paris in 2019"); ok {
		t.Fatal("another session must not see this session's answers")
	}
}

func TestGetCitationGuidelinesUsesDefaultWithoutOverride(t *testing.T) {
	got := GetCitationGuidelines("")
	// The default is the embedded citation_prompt.md (mirrors Python
	// load_prompt("citation_prompt")). It must contain the citation-rules body.
	if !strings.Contains(got, "# Citation Requirements:") ||
		!strings.Contains(got, "Place citations at the end of sentences") {
		t.Fatalf("got %q, want the default citation rules", got)
	}
}

func TestGetCitationGuidelinesHonoursOverride(t *testing.T) {
	// Python renders the user's template; Go takes the rendered result verbatim.
	got := GetCitationGuidelines("Cite as [n].")
	if got != "Cite as [n]." {
		t.Fatalf("got %q, want the override", got)
	}
}

func TestSysPromptIncludesSummarizeOnlyWithUnstructured(t *testing.T) {
	withDoc := SysPrompt("", true)
	if !strings.Contains(withDoc, "summarize_document") {
		t.Fatal("summarize_document must be offered when unstructured retrieval exists")
	}
	withoutDoc := SysPrompt("", false)
	if strings.Contains(withoutDoc, "summarize_document") {
		t.Fatal("summarize_document must be omitted without unstructured retrieval")
	}
}

func TestSysPromptPrependsSystemPrompt(t *testing.T) {
	got := SysPrompt("You are helpful.", true)

	if !strings.HasPrefix(got, "You are helpful.\n\n") {
		t.Fatalf("got %q, want the system prompt first", got)
	}
	if !strings.Contains(got, "call the `rag` tool") {
		t.Fatalf("got %q, want the router body kept", got)
	}
}

func TestFitEvidenceKeepsShortEvidence(t *testing.T) {
	evidence := "short evidence"
	if got := FitEvidence("q", evidence); got != evidence {
		t.Fatalf("got %q, want %q unchanged", got, evidence)
	}
}

func TestFitEvidenceReturnsEmptyForEmpty(t *testing.T) {
	if got := FitEvidence("q", ""); got != "" {
		t.Fatalf("got %q, want empty", got)
	}
}

func TestFitEvidenceTrimsOversizedEvidence(t *testing.T) {
	// The budget is fixed (not the model window), so a large pool can never fill
	// the context — evidence beyond the cap must be dropped.
	evidence := strings.Repeat("word ", 200000)
	got := FitEvidence("q", evidence)

	if len(got) >= len(evidence) {
		t.Fatalf("evidence was not trimmed: %d >= %d", len(got), len(evidence))
	}
	if got == "" {
		t.Fatal("evidence must not be trimmed to nothing")
	}
}

func TestGraphRecursionLimitMatchesPython(t *testing.T) {
	// Python :1619 — 60 for the agentic graph, else max(25, max_loops*8).
	if got := graphRecursionLimit(true, 3); got != 60 {
		t.Fatalf("agentic limit = %d, want 60", got)
	}
	if got := graphRecursionLimit(false, 3); got != 25 {
		t.Fatalf("low limit with 3 loops = %d, want 25", got)
	}
	if got := graphRecursionLimit(false, 5); got != 40 {
		t.Fatalf("low limit with 5 loops = %d, want 40", got)
	}
}

func TestAgenticResearchRoundCostsThreeVisits(t *testing.T) {
	// One research round is rag_agent → draft → sca. If the loop counted
	// iterations instead of node visits, 20 rounds would cost 20 instead of 60
	// and the guard would trip ~3x later than Python's.
	if got, want := agenticRoundVisits, 3; got != want {
		t.Fatalf("round cost = %d, want %d", got, want)
	}
}

func TestNewAgenticStateDoesNotArmBudget(t *testing.T) {
	// Python :823 — the budget is armed in the formalize_question node's
	// return, not at state creation, so formalization is not charged to it.
	st := NewAgenticState("q", "", 3, nil)

	if !st.Deadline.IsZero() {
		t.Fatalf("Deadline = %v, want zero until formalize_question runs", st.Deadline)
	}
}

func TestFormalizeQuestionNodeArmsBudget(t *testing.T) {
	st := NewAgenticState("when did it open?", "", 3, nil)
	// No model: the node still arms the budget, mirroring Python :823.
	formalizeQuestionNode(context.Background(), RAGTools{}, st, nil)

	if st.Deadline.IsZero() {
		t.Fatal("formalize_question must arm the budget (Python :823)")
	}
	if st.Question != "when did it open?" {
		t.Fatalf("Question = %q, want it unchanged without a model", st.Question)
	}
}

func TestSessionCacheEmptySessionIsIsolated(t *testing.T) {
	// Without a session there is nothing to scope by, so each call gets its own
	// cache rather than one shared by every conversation.
	a := SessionCache("")
	b := SessionCache("")
	a.Store("population of Paris 2019", "2.1 million")

	if _, ok := b.Lookup("legal population of Paris in 2019"); ok {
		t.Fatal("an unscoped cache must not leak across calls")
	}
}
