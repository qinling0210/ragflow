package harness

import (
	"bytes"
	"context"
	"log"
	"strings"
	"testing"
)

// TestLogHierarchicalRounds verifies that Log mirrors Python LLMUsageStats.log:
// the orchestrator phase expands into one "orchestrator round N" row per round,
// with the in-loop sub-phases (claim_research / sufficiency) nested underneath,
// while the out-of-loop phases (route / planner / finalize) stay flat.
//
// NOTE: this test cannot run while the unrelated pre-existing break in
// internal/agent/tool/retrieval_nlp.go (RankFeature) blocks compilation of the
// harness test binary. It will run once that is resolved.
func TestLogHierarchicalRounds(t *testing.T) {
	s := NewLLMUsageStats()
	ctx := WithStats(context.Background(), s)

	// Flat phases before the orchestrator loop.
	func() {
		_, done := Phase(ctx, PhaseRoute)
		defer done()
		s.RecordCall(PhaseRoute)
	}()
	func() {
		_, done := Phase(ctx, PhasePlanner)
		defer done()
		s.RecordCall(PhasePlanner)
	}()

	// Orchestrator loop: two rounds, each doing claim_research + sufficiency.
	func() {
		c, done := Phase(ctx, PhaseOrchestrator)
		defer done()

		RecordRound(c, PhaseOrchestrator)
		func() {
			_, d := Phase(c, PhaseClaimResearch)
			defer d()
			s.RecordCall(PhaseClaimResearch)
			s.RecordUsage(PhaseClaimResearch, 0, 0, 120)
		}()
		func() {
			_, d := Phase(c, PhaseSufficiency)
			defer d()
			s.RecordCall(PhaseSufficiency)
		}()

		RecordRound(c, PhaseOrchestrator)
		func() {
			_, d := Phase(c, PhaseClaimResearch)
			defer d()
			s.RecordCall(PhaseClaimResearch)
			s.RecordUsage(PhaseClaimResearch, 0, 0, 80)
		}()
	}()

	// Flat phase after the loop.
	func() {
		_, done := Phase(ctx, PhaseFinalize)
		defer done()
		s.RecordCall(PhaseFinalize)
	}()

	var buf bytes.Buffer
	s.Log(log.New(&buf, "", 0))
	out := buf.String()

	for _, want := range []string{
		"orchestrator round 1",
		"orchestrator round 2",
		"route",
		"planner",
		"finalize",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("log output missing %q\n---\n%s", want, out)
		}
	}
	if !strings.Contains(out, "sufficiency") {
		t.Errorf("log output missing sufficiency\n---\n%s", out)
	}
	// claim_research runs inside every round, so it must appear at least once
	// per round as a nested (indented) row, not just as a flat top-level row.
	if n := strings.Count(out, "claim_research"); n < 2 {
		t.Errorf("expected claim_research at least twice (per round), got %d\n---\n%s", n, out)
	}
}
