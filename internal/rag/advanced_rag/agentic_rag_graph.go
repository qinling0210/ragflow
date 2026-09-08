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

// Package advanced_rag is the outer agentic-search loop (medium / high / ultra).
//
// This file mirrors Python rag/advanced_rag/agentic_rag_graph.py — the
// five-phase pipeline that sits ABOVE the action session:
//
//	formalize_question → [planner → prefetch] → rag_agent → draft → sca
//	    ├─ sufficient ──────────────────────────→ formalize_answer
//	    └─ insufficient → query_rewrite ────────→ rag_agent (next round)
//
// PYTHON USES LANGGRAPH; THIS USES AN EXPLICIT LOOP, matching the choice made
// for the action session. The graph is small and fixed, so the node bodies,
// routing predicates, and their ordering are ported verbatim and only the
// driver differs.
//
// The RAGTools-configured dependencies live in agentic_rag.go, which
// mirrors Python rag/advanced_rag/agentic_rag.py. This split replicates the
// Python layout: agentic_rag.py (RAGTools) is a sibling of agentic_rag_graph.py,
// and both sit at the same level as harness/.
package advanced_rag

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"reflect"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/cloudwego/eino/schema"

	"ragflow/internal/agent/chat"
	"ragflow/internal/common"
	"ragflow/internal/rag/advanced_rag/harness"
	"ragflow/internal/rag/advanced_rag/harness/orchestrator"
	"ragflow/internal/rag/prompts"
)

// ---------------------------------------------------------------------------
// Global research budget & per-call timeouts (Python lines 75-81).
//
// The benchmark client cuts a request at 300s (read timeout). These bounds keep
// one question's WHOLE pipeline comfortably under that line: when the budget
// runs out the routing guards steer to synthesis with whatever evidence is on
// hand instead of starting another research round.
// ---------------------------------------------------------------------------

var _LOG = common.StdLogger()

const (
	TotalBudgetS      = 180.0 // whole-graph wall-clock ceiling per question
	MinRoundHeadroomS = 50.0  // need at least this much left to start a new round
	PassTimeoutS      = 120.0 // slot research pass wall-clock
	PrefetchTimeoutS  = 90.0  // programmatic fan-out fetch
	DraftTimeoutS     = 60.0  // fallback draft synthesis
	SCATimeoutS       = 60.0  // sufficient-context review call
	RewriteTimeoutS   = 45.0  // gap → query rewrite call
	// SCAViewCap: 24 of 225 hid the answer-bearing table chunk from the SCA.
	SCAViewCap = 60
	// MaxSnippetPool is the storage ceiling of the snippet pool across ALL
	// rounds. Storage and REVIEW are decoupled: the SCA only reads a ranked
	// view, so the pool may accumulate freely while prompts stay bounded.
	MaxSnippetPool = 60
	// DrillReserve: slots kept free after the FIRST prefetch so the research
	// executor can top up evidence.
	DrillReserve = 12
	// FanoutTopN is the per-query result count for the programmatic fetch.
	FanoutTopN = 8
	// FanoutTopNRewrite is the reduced count used after a rewrite round.
	FanoutTopNRewrite = 6
	// MaxFanouts caps planner fan-outs (Python: [:5]).
	MaxFanouts = 5
	// MaxSCAGaps caps gaps handed to the rewriter (Python: gaps[:8]).
	MaxSCAGaps = 8
	// PoolHeadLines caps the evidence-pool summary shown to the rewriter
	// (Python: chunks[:12]).
	PoolHeadLines = 12

	// Slot-table research constants (Python: asyncio.Semaphore(2), [:3], etc.).
	slotSessionConcurrency = 2
	slotSessionsPerRound   = 3
	slotFallbackClueChars  = 160
	draftCandidateChars    = 400
	draftClueChars         = 200
)

// Verdict statuses.
const (
	VerdictSufficient   = "SUFFICIENT"
	VerdictInsufficient = "INSUFFICIENT"
)

// ---------------------------------------------------------------------------
// Local text helpers (mirror the helpers Python defines in agentic_rag_graph.py
// before AgenticState: _snip / _safe_list / _is_poisoned / _view_terms / ...).
// ---------------------------------------------------------------------------

var tokenPattern = regexp.MustCompile(`[A-Za-z0-9_]+`)

// queryToTerms mirrors the harness keywords helper used by Python _view_terms:
// lowercase word tokens of length >= 3, de-duplicated, order preserved.
func queryToTerms(q string) []string {
	if q == "" {
		return nil
	}
	found := tokenPattern.FindAllString(strings.ToLower(q), -1)
	out := make([]string, 0, len(found))
	seen := make(map[string]bool, len(found))
	for _, t := range found {
		if len(t) < 3 || seen[t] {
			continue
		}
		seen[t] = true
		out = append(out, t)
	}
	return out
}

// truncateRunes caps a string to n runes without breaking multi-byte chars.
func truncateRunes(s string, n int) string {
	if n <= 0 {
		return ""
	}
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n])
}

// dedupe preserves order and drops empty / repeated entries.
func dedupe(in []string) []string {
	if len(in) == 0 {
		return nil
	}
	seen := make(map[string]bool, len(in))
	out := make([]string, 0, len(in))
	for _, s := range in {
		if s == "" || seen[s] {
			continue
		}
		seen[s] = true
		out = append(out, s)
	}
	return out
}

// stringOf coerces an arbitrary JSON value to a string (Python str()).
func stringOf(v any) string {
	switch x := v.(type) {
	case nil:
		return ""
	case string:
		return x
	case fmt.Stringer:
		return x.String()
	default:
		return fmt.Sprint(x)
	}
}

// asSliceOfAny coerces a JSON array value to []any (Python _safe_list, line 95).
//
// Python also guards against a coroutine reaching a state field under async
// concurrency; Go cannot hit that case, but isPoisoned covers the equivalents
// that equally must never reach a prompt (see its doc comment).
func asSliceOfAny(v any) []any {
	if isPoisoned(v) {
		_LOG.Printf("[StateGuard] dropping poisoned value of kind %s; treating as empty",
			reflect.ValueOf(v).Kind())
		return nil
	}
	switch x := v.(type) {
	case nil:
		return nil
	case []any:
		return x
	case []string:
		out := make([]any, len(x))
		for i, s := range x {
			out[i] = s
		}
		return out
	case map[string]any:
		out := make([]any, 0, len(x))
		for _, vv := range x {
			out = append(out, vv)
		}
		return out
	default:
		return []any{v}
	}
}

// toIntStrict parses an int-like value (JSON numbers arrive as float64).
func toIntStrict(v any) (int, bool) {
	switch x := v.(type) {
	case int:
		return x, true
	case int64:
		return int(x), true
	case float64:
		return int(x), true
	case string:
		var n int
		if _, err := fmt.Sscanf(x, "%d", &n); err == nil {
			return n, true
		}
	}
	return 0, false
}

// toFloat parses a numeric value.
func toFloat(v any) (float64, bool) {
	switch x := v.(type) {
	case float64:
		return x, true
	case int:
		return float64(x), true
	case int64:
		return float64(x), true
	case string:
		var f float64
		if _, err := fmt.Sscanf(x, "%f", &f); err == nil {
			return f, true
		}
	}
	return 0, false
}

// anyString reads a string field from a chunk-like map.
func anyString(v any) string {
	switch x := v.(type) {
	case nil:
		return ""
	case string:
		return x
	default:
		return fmt.Sprint(x)
	}
}

// truncateEach caps every entry of a string slice.
func truncateEach(in []string, n int) []string {
	if len(in) == 0 {
		return nil
	}
	out := make([]string, len(in))
	for i, s := range in {
		out[i] = truncateRunes(s, n)
	}
	return out
}

// equalStringPtr reports whether two optional strings are equal.
func equalStringPtr(a, b *string) bool {
	if a == nil || b == nil {
		return a == b
	}
	return *a == *b
}

// equalStrings reports set-equality (order-independent) of two string slices.
func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	seen := make(map[string]int, len(a))
	for _, s := range a {
		seen[s]++
	}
	for _, s := range b {
		if seen[s] == 0 {
			return false
		}
		seen[s]--
	}
	return true
}

// isPoisoned mirrors Python _is_poisoned (line 116): true when a state value
// must be discarded before it poisons an LLM prompt.
//
// Python guards against coroutine objects leaking into graph-state fields under
// async concurrency. Go's type system makes that specific leak impossible, so
// this checks the Go equivalents that must never reach a prompt: channels and
// function values, neither of which renders as anything a model can read.
func isPoisoned(v any) bool {
	if v == nil {
		return false
	}
	switch reflect.ValueOf(v).Kind() {
	case reflect.Chan, reflect.Func, reflect.UnsafePointer:
		return true
	}
	return false
}

// SelectSCAView mirrors Python _select_sca_view (line 121): rank the stored pool
// down to the SCA review view (storage ≠ review).
//
// Score = retrieval relevance + surface-term coverage + freshness bonus for
// chunks admitted in later rounds. Returns (view, identity) where identity is a
// stable hash of the selected chunk ids — the caller uses it to detect an
// unproductive round (same view twice despite new storage ⇒ nothing new).
func SelectSCAView(chunks []map[string]any, focusTerms []string) ([]map[string]any, string) {
	terms := dedupe(focusTerms)
	lowered := make([]string, 0, len(terms))
	for _, t := range terms {
		if len(t) >= 3 {
			lowered = append(lowered, strings.ToLower(t))
		}
	}
	type scored struct {
		idx   int
		chunk map[string]any
		score float64
	}
	ranked := make([]scored, 0, len(chunks))
	for i, c := range chunks {
		text := strings.ToLower(strings.Join([]string{
			anyString(c["content"]), anyString(c["content_with_weight"]),
			anyString(c["title"]), anyString(c["question_toks"]),
		}, " "))
		cov := 0
		for _, t := range lowered {
			if strings.Contains(text, t) {
				cov++
			}
		}
		covRatio := 0.5
		if len(lowered) > 0 {
			covRatio = float64(cov) / float64(len(lowered))
		}
		rel := 0.0
		if v, ok := toFloat(c["similarity"]); ok {
			rel = v
		} else if v, ok := toFloat(c["score"]); ok {
			rel = v
		}
		fresh := min(float64(i)/20.0, 0.2) // late arrivals (gap-pursuit evidence) get seen
		ranked = append(ranked, scored{i, c, rel*0.45 + min(covRatio, 1.0)*0.45 + fresh})
	}
	sort.SliceStable(ranked, func(i, j int) bool { return ranked[i].score > ranked[j].score })
	limit := min(len(ranked), SCAViewCap)
	view := make([]map[string]any, 0, limit)
	ids := make([]string, 0, limit)
	for _, r := range ranked[:limit] {
		view = append(view, r.chunk)
		ids = append(ids, harness.ChunkIDOf(r.chunk))
	}
	sort.Strings(ids)
	return view, joinHash(ids)
}

// joinHash builds a stable identity from the selected chunk ids. Mirrors
// Python's `str(hash("|".join(sorted(ids))))` in spirit (a deterministic digest
// rather than Python's randomised-per-process hash).
func joinHash(ids []string) string {
	return fmt.Sprintf("%x", strings.Join(ids, "|"))
}

// ViewTerms mirrors Python _view_terms (line 150): terms describing what the SCA
// should look FOR this round.
func ViewTerms(st *AgenticState) []string {
	if st == nil {
		return nil
	}
	terms := queryToTerms(st.Question)
	for _, q := range st.CurrentQueries {
		terms = append(terms, queryToTerms(q)...)
	}
	return dedupe(terms)
}

// RemainingS mirrors Python _remaining_s: seconds left in the global budget.
// Never negative.
func (s *AgenticState) RemainingS() float64 {
	if s.Deadline.IsZero() {
		return TotalBudgetS
	}
	d := time.Until(s.Deadline).Seconds()
	if d < 0 {
		return 0
	}
	return d
}

// bounded mirrors Python _bounded (line 167): run fn under a wall-clock bound.
// On expiry it logs and returns the zero value, so one slow step never stalls
// the whole question. A non-positive bound means "no bound".
//
// The pipeline nodes (prefetch / research pass / draft / SCA / rewrite) wrap
// their calls by hand instead of routing through this helper, because each
// needs to record per-node state on expiry (partial flags, round counters) —
// not merely fall back to a zero value. This helper covers the plain case and
// is the shape those hand-written guards shrink to once they stop needing the
// extra bookkeeping.
func bounded[T any](ctx context.Context, timeoutS float64, what string, fn func(context.Context) (T, error)) (T, error) {
	var zero T
	if timeoutS <= 0 {
		return fn(ctx)
	}
	runCtx, cancel := context.WithTimeout(ctx, time.Duration(timeoutS*float64(time.Second)))
	defer cancel()
	out, err := fn(runCtx)
	if errors.Is(err, context.DeadlineExceeded) {
		_LOG.Printf("[Budget] %s exceeded %.0fs — moving on without it", what, timeoutS)
		return zero, nil
	}
	return out, err
}

// ---------------------------------------------------------------------------
// AgenticState — the outer loop's mutable state (Python AgenticState, line 177).
// ---------------------------------------------------------------------------

type AgenticState struct {
	// ── conversation input ──
	// Messages is the conversation history the formalize_question node reads to
	// resolve pronouns and ellipses (Python: messages).
	Messages []schema.Message
	Question string
	Keywords string

	// ── evolving research state ──
	Plan            []string // planner fan-outs (Phase 1)
	CurrentQueries  []string // active research targets
	SlotTable       harness.State
	SlotDraft       string // slot-rendered fact draft fed to the SCA
	CollectedAnswer string // non-terminal <answer> candidate for SCA validation
	UnresolvedSlots []map[string]any
	SlotEvidence    map[string]SlotEvidence
	KB              *harness.Kbinfos
	Draft           string // intermediate fact-preserving draft (Phase 3 reviewee)
	RagAnswer       string
	PartialAnswer   bool
	Abstain         bool
	EmptyResult     bool
	Verdict         string // VerdictSufficient / VerdictInsufficient
	SCA             map[string]any

	// ── budgets & counters ──
	MaxLoops     int
	Deadline     time.Time // wall-clock expiry of the research budget
	SearchRounds int       // completed SCA→query_rewrite iterations
	SCAViewID    string    // identity of the last SCA review view
	Attempted    []map[string]any
	NoProgress   bool
}

// NewAgenticState builds the initial state. It does NOT arm the global budget:
// Python sets `deadline` inside the formalize_question node's return (:823), so
// formalization is not charged to it. Mirroring that keeps the deadline's owner
// identical to Python's.
//
// maxLoops mirrors Python's max_loops.
func NewAgenticState(question, keywords string, maxLoops int, messages []schema.Message) *AgenticState {
	if maxLoops <= 0 {
		maxLoops = 3
	}
	return &AgenticState{
		Question:    question,
		Keywords:    keywords,
		MaxLoops:    maxLoops,
		Messages:    messages,
		KB:          &harness.Kbinfos{},
		EmptyResult: true,
	}
}

// ---------------------------------------------------------------------------
// Thinking-tag stream splitting (Python lines 223-292).
// ---------------------------------------------------------------------------

// Thinking-tag delimiters (Python _THINK_OPEN / _THINK_CLOSE, lines 223-224).
const (
	thinkOpen  = "<think>"
	thinkClose = "</think>"
)

// partialTagTail mirrors Python _partial_tag_tail (line 227): the length of the
// longest proper prefix of tag that s ends with — how much of a tag may still
// be arriving on the next stream delta.
func partialTagTail(s, tag string) int {
	limit := len(s)
	if max := len(tag) - 1; max < limit {
		limit = max
	}
	for k := limit; k > 0; k-- {
		if strings.HasSuffix(s, tag[:k]) {
			return k
		}
	}
	return 0
}

// ThinkChunk is one split-out piece of a model stream: either reasoning
// ("think") or user-visible text ("answer").
type ThinkChunk struct {
	Kind string // "think" | "answer"
	Text string
}

// SplitThinkStream mirrors Python _split_think_stream (line 234): split model
// deltas into "think" and "answer" text.
//
// Besides ordinary <think>...</think> streams, some providers emit the opening
// tag only on the first reasoning delta and append </think> to every
// subsequent delta, so an unmatched closing tag still marks the text before it
// as reasoning.
//
// Python is an async generator; Go returns a channel that is closed once deltas
// are drained or ctx is cancelled.
func SplitThinkStream(ctx context.Context, deltas <-chan string) <-chan ThinkChunk {
	out := make(chan ThinkChunk)
	go func() {
		defer close(out)
		var buf strings.Builder
		inThink := false

		emit := func(kind, text string) bool {
			if text == "" {
				return true
			}
			select {
			case out <- ThinkChunk{Kind: kind, Text: text}:
				return true
			case <-ctx.Done():
				return false
			}
		}

		for {
			var token string
			var ok bool
			select {
			case token, ok = <-deltas:
				if !ok {
					// Flush the tail, stripping any residual tag.
					if rest := buf.String(); rest != "" {
						rest = strings.ReplaceAll(rest, thinkOpen, "")
						rest = strings.ReplaceAll(rest, thinkClose, "")
						kind := "answer"
						if inThink {
							kind = "think"
						}
						emit(kind, rest)
					}
					return
				}
			case <-ctx.Done():
				return
			}

			buf.WriteString(token)
			s := buf.String()

			for s != "" {
				if inThink {
					closeIdx := strings.Index(s, thinkClose)
					if closeIdx >= 0 {
						if !emit("think", s[:closeIdx]) {
							return
						}
						s = s[closeIdx+len(thinkClose):]
						inThink = false
						continue
					}
					if hold := partialTagTail(s, thinkClose); hold > 0 {
						if !emit("think", s[:len(s)-hold]) {
							return
						}
						s = s[len(s)-hold:]
					} else {
						if !emit("think", s) {
							return
						}
						s = ""
					}
					break
				}

				openIdx := strings.Index(s, thinkOpen)
				closeIdx := strings.Index(s, thinkClose)

				if closeIdx >= 0 && (openIdx < 0 || closeIdx < openIdx) {
					if !emit("think", s[:closeIdx]) {
						return
					}
					s = s[closeIdx+len(thinkClose):]
					continue
				}
				if openIdx >= 0 {
					if !emit("answer", s[:openIdx]) {
						return
					}
					s = s[openIdx+len(thinkOpen):]
					inThink = true
					continue
				}

				hold := partialTagTail(s, thinkOpen)
				if h := partialTagTail(s, thinkClose); h > hold {
					hold = h
				}
				if hold > 0 {
					if !emit("answer", s[:len(s)-hold]) {
						return
					}
					s = s[len(s)-hold:]
				} else {
					if !emit("answer", s) {
						return
					}
					s = ""
				}
				break
			}

			buf.Reset()
			buf.WriteString(s)
		}
	}()
	return out
}

// SCAGapsToRewrite mirrors Python _sca_gaps_to_rewrite (line 298).
//
// Preference order:
//  1. Unsatisfied sub_queries (the precise "what is missing / where to search
//     next" signal from Q-CARE) — (missing_fact, search_hint).
//  2. Per-claim missing_information items — (what, search_hint).
//
// Returns [(what, search_hint), ...] (deduped, non-empty).
func SCAGapsToRewrite(sca map[string]any) []orchestrator.MissingPiece {
	var gaps []orchestrator.MissingPiece
	seen := map[string]bool{}
	add := func(what, hint string) {
		what = strings.TrimSpace(what)
		hint = strings.TrimSpace(hint)
		if what == "" && hint == "" {
			return
		}
		key := what + "|" + hint
		if seen[key] {
			return
		}
		seen[key] = true
		if what == "" {
			what = hint
		}
		if hint == "" {
			hint = what
		}
		gaps = append(gaps, orchestrator.MissingPiece{What: what, SearchHint: hint})
	}
	if raw, ok := sca["sub_queries"].([]any); ok {
		for _, item := range raw {
			m, ok := item.(map[string]any)
			if !ok {
				continue
			}
			if b, ok := m["satisfied"].(bool); ok && b {
				continue
			}
			what := stringOf(m["missing_fact"])
			if what == "" {
				what = stringOf(m["sub_query"])
			}
			add(what, stringOf(m["search_hint"]))
		}
	}
	if len(gaps) == 0 {
		// Fall back to per-claim missing_information.
		switch claims := sca["claims"].(type) {
		case map[string]orchestrator.ClaimVerdict:
			keys := make([]string, 0, len(claims))
			for k := range claims {
				keys = append(keys, k)
			}
			sort.Strings(keys)
			for _, k := range keys {
				for _, mi := range claims[k].MissingInformation {
					add(mi.What, mi.SearchHint)
				}
			}
		case map[string]any:
			for _, g := range claims {
				if gm, ok := g.(map[string]any); ok {
					for _, mi := range asSliceOfAny(gm["missing_information"]) {
						if mm, ok := mi.(map[string]any); ok {
							add(stringOf(mm["what"]), stringOf(mm["search_hint"]))
						} else {
							add(stringOf(mi), "")
						}
					}
				}
			}
		}
	}
	if len(gaps) > MaxSCAGaps {
		gaps = gaps[:MaxSCAGaps]
	}
	return gaps
}

// ---------------------------------------------------------------------------
// Phase 1: fan-out expansion (Python _expand_fanouts, line 382).
// ---------------------------------------------------------------------------

const fanoutPrompt = `Break the user's question into 2 to 5 independent, directly searchable sub-questions (fan-outs). Each must be self-contained enough to retrieve relevant passages from a document corpus on its own. For multi-hop questions, produce ONLY the first-hop sub-questions needed to start (the anchor facts); do not invent downstream hops that depend on answers you do not have yet.
Respond with a JSON object: {"fanouts": ["...", "..."]}. No prose, JSON only.`

// extractJSONObject mirrors Python _extract_json_object (line 350): return the
// first parseable JSON object in text, or nil when none parses.
//
// The fan-out model sometimes emits prose around the object, and a greedy
// brace-match would capture several objects and fail with "extra data"; each
// candidate is therefore validated before it is accepted, and an invalid one
// resumes the scan at its next "{".
func extractJSONObject(text string) any {
	for i := 0; i < len(text); {
		rel := strings.IndexByte(text[i:], '{')
		if rel < 0 {
			return nil
		}
		start := i + rel
		depth := 0
	scan:
		for j := start; j < len(text); j++ {
			switch text[j] {
			case '{':
				depth++
			case '}':
				depth--
				if depth == 0 {
					var out map[string]any
					if json.Unmarshal([]byte(text[start:j+1]), &out) == nil {
						return out
					}
					break scan // not a valid object; try the next "{"
				}
			}
		}
		i = start + 1
	}
	return nil
}

// ExpandFanouts mirrors Python _expand_fanouts (line 382): ONE chat call (no
// tools) producing 2-5 first-hop fan-outs.
//
// Falls back to the raw question alone on any failure — a fan-out failure never
// blocks the pipeline.
func ExpandFanouts(ctx context.Context, deps RAGTools, question string) []string {
	if question == "" {
		return nil
	}
	if deps.Model == nil {
		return []string{question}
	}
	reply, err := deps.Model.Complete(ctx, []schema.Message{
		*schema.SystemMessage(fanoutPrompt),
		*schema.UserMessage("Question: " + question),
	}, nil)
	if err != nil {
		_LOG.Printf("[Planner] fan-out expansion failed; falling back to raw question: %v", err)
		return []string{question}
	}
	if data, ok := extractJSONObject(reply.Content).(map[string]any); ok {
		fanouts := make([]string, 0, MaxFanouts)
		for _, f := range asSliceOfAny(data["fanouts"]) {
			if s := strings.TrimSpace(fmt.Sprint(f)); s != "" {
				fanouts = append(fanouts, s)
			}
		}
		fanouts = dedupe(fanouts)
		if len(fanouts) > MaxFanouts {
			fanouts = fanouts[:MaxFanouts]
		}
		if len(fanouts) > 0 {
			_LOG.Printf("[Planner] fan-out expansion: %d sub-question(s): %v", len(fanouts), fanouts)
			return fanouts
		}
	}
	// Fallback: split on line items from a loose answer.
	fanouts := make([]string, 0, MaxFanouts)
	for _, ln := range strings.Split(reply.Content, "\n") {
		ln = strings.TrimSpace(strings.TrimLeft(ln, "-•0123456789. "))
		if ln != "" {
			fanouts = append(fanouts, ln)
		}
	}
	fanouts = dedupe(fanouts)
	if len(fanouts) > MaxFanouts {
		fanouts = fanouts[:MaxFanouts]
	}
	if len(fanouts) == 0 {
		return []string{question}
	}
	return fanouts
}

// ---------------------------------------------------------------------------
// Phase 2: programmatic fan-out search (Python _fanout_search, line 432).
// ---------------------------------------------------------------------------

// FanoutSearch mirrors Python _fanout_search (line 432): fetch every query and
// fill the pool. Returns how many NEW chunks were added; per-query failures are
// swallowed.
//
// The Go port uses the single hybrid path (see harness.SearchFn) rather than
// Python's two collector channels (BM25 + narrow, and hybrid bypass), because
// the Go Retriever interface exposes one weighted call. The narrow step is the
// all-or-nothing form (see harness.NarrowOrKeep), so semantic hits sharing no
// surface words survive rather than being filtered out.
//
// The executor is responsible for merging into st.KB; this returns the delta.
func FanoutSearch(ctx context.Context, deps RAGTools, st *AgenticState, queries []string, topN, capacity int) int {
	if deps.Tools == nil || deps.Tools.Exec == nil || len(queries) == 0 {
		return 0
	}
	before := len(st.KB.Chunks)
	for _, q := range queries {
		select {
		case <-ctx.Done():
			return len(st.KB.Chunks) - before
		default:
		}
		if _, err := deps.Tools.Exec.Execute(ctx, "search_chunks", map[string]any{
			"query": []string{q},
		}); err != nil {
			_LOG.Printf("[FanoutSearch] query %q failed: %v", truncateRunes(q, 60), err)
			continue
		}
		if capacity > 0 && len(st.KB.Chunks) >= capacity {
			break
		}
	}
	return len(st.KB.Chunks) - before
}

// BuildLowGraph mirrors Python build_low_graph (line 733): the lightweight
// low-mode path — formalize → direct_search → answer — with no planner, no
// fan-out, and no SCA loop.
//
// Python compiles a three-node LangGraph and run_agentic_rag invokes it (:1611).
// Go has no graph runtime and therefore no compile step: this runs those nodes
// directly, so it both "builds" and "runs" the graph. The name follows Python's
// build_low_graph for traceability.
func BuildLowGraph(ctx context.Context, deps RAGTools, req harness.RunRequest, sd harness.SearchDeps, kb *harness.Kbinfos, resp *RunResponse, logger *log.Logger) {
	// Node 1: formalize_question (:738), then node 2: direct_search (:595-602).
	formalizeQuestion(ctx, deps, &req, logger)
	runDirect(ctx, deps, req, sd, kb, resp, logger)
}

// ---------------------------------------------------------------------------
// Routing
// ---------------------------------------------------------------------------

type agenticNode int

const (
	// nodeFormalizeQuestion is the graph's entry node (Python
	// build_agentic_graph: `add_edge(START, "formalize_question")`).
	nodeFormalizeQuestion agenticNode = iota
	nodeFormalizeAnswer
	nodeQueryRewrite
	nodeRagAgentLoop
	nodePlanner
	nodePrefetch
	nodeRagAgentFirst
)

// plannerNode mirrors the `planner` node (line 828): decompose into fan-outs,
// then build the slot table from them.
func plannerNode(ctx context.Context, deps RAGTools, st *AgenticState, logger *log.Logger) {
	ctx, done := harness.Phase(ctx, "planner")
	defer done()

	logger.Printf("[Planner] Decomposing the question into first-hop fan-outs...")
	fanouts := ExpandFanouts(ctx, deps, st.Question)
	st.Plan = fanouts
	st.CurrentQueries = append([]string(nil), fanouts...)

	// Build the slot table right after fan-out decomposition so the research
	// pass is slot-directed (each slot = one unknown to resolve).
	sd := deps.sessionDeps()
	root, firstQueries := BuildSlotTable(ctx, sd, st.Question, fanouts, max(15.0, st.RemainingS()-15.0))
	st.SlotTable = root
	if len(firstQueries) > 0 {
		st.CurrentQueries = firstQueries
	}
}

// prefetchNode mirrors the `prefetch` node (line 850): programmatic fan-out
// retrieval into the snippet pool.
func prefetchNode(ctx context.Context, deps RAGTools, st *AgenticState, logger *log.Logger, firstRound bool) {
	ctx, done := harness.Phase(ctx, "orchestrator")
	defer done()

	queries := st.CurrentQueries
	if len(queries) == 0 {
		if st.Question == "" {
			return
		}
		queries = []string{st.Question}
	}
	timeout := min(PrefetchTimeoutS, max(10.0, st.RemainingS()-MinRoundHeadroomS))
	callCtx, cancel := context.WithTimeout(ctx, time.Duration(timeout*float64(time.Second)))
	defer cancel()

	capacity := MaxSnippetPool
	if firstRound {
		// First-round prefetch leaves drill slots free.
		capacity = MaxSnippetPool - DrillReserve
	}
	added := FanoutSearch(callCtx, deps, st, queries, FanoutTopN, capacity)
	for _, q := range queries {
		st.Attempted = append(st.Attempted, map[string]any{"q": q, "r": 0, "new": added})
	}
}

// ragAgentNode mirrors the `rag_agent` node (line 876): one slot research pass.
func ragAgentNode(ctx context.Context, deps RAGTools, st *AgenticState, logger *log.Logger) {
	ctx, done := harness.Phase(ctx, "dynamic")
	defer done()

	timeLeft := st.RemainingS()
	if timeLeft < MinRoundHeadroomS {
		logger.Printf("[RAGAgent] only %.0fs left of the research budget; skipping further passes.", timeLeft)
		return
	}
	roundNo := st.SearchRounds + 1
	poolBefore := len(st.KB.Chunks)
	logger.Printf("[RAGAgent] ROUND %d start (search_rounds=%d, time_left=%.0fs, pool=%d chunks)",
		roundNo, st.SearchRounds, timeLeft, poolBefore)

	t := max(20.0, min(PassTimeoutS, timeLeft-25.0))
	callCtx, cancel := context.WithTimeout(ctx, time.Duration(t*float64(time.Second)))
	defer cancel()

	res := RunSlotResearchPass(callCtx, deps.sessionDeps(), st.Question, st, t)
	if res == nil {
		return
	}
	st.SlotTable = res.SlotTable
	st.CollectedAnswer = res.CollectedAnswer
	st.UnresolvedSlots = res.UnresolvedSlots
	st.SlotEvidence = res.SlotEvidence
	st.SlotDraft = res.SlotDraft
	st.RagAnswer = res.SlotDraft
	if st.RagAnswer == "" {
		st.RagAnswer = st.SlotDraft
	}
	st.Attempted = res.Attempted

	logger.Printf("[RAGAgent] ROUND %d end (+%d new chunks, pool=%d, unresolved=%d)",
		roundNo, len(st.KB.Chunks)-poolBefore, len(st.KB.Chunks), len(res.UnresolvedSlots))
}

// draftNode mirrors the `draft` node (line 920): the intermediate draft that the
// SCA reviews.
func draftNode(ctx context.Context, deps RAGTools, st *AgenticState, logger *log.Logger) {
	ctx, done := harness.Phase(ctx, "draft")
	defer done()

	draftText := strings.TrimSpace(st.RagAnswer)
	if draftText == "" {
		// Budget exhaustion without a report: synthesize one from snippets.
		t := min(DraftTimeoutS, max(15.0, st.RemainingS()-10.0))
		callCtx, cancel := context.WithTimeout(ctx, time.Duration(t*float64(time.Second)))
		draftText = ComposeFallbackDraft(callCtx, deps, st)
		cancel()
	}
	if draftText != "" {
		st.KB.PreSummary = draftText
	}
	st.Draft = draftText
	logger.Printf("[Draft] intermediate draft %d chars (evidence=%d chunks)", len(draftText), len(st.KB.Chunks))
}

// scaNode mirrors the `sca` node (line 941): the Phase-3 quality-control review.
func scaNode(ctx context.Context, deps RAGTools, st *AgenticState, logger *log.Logger) {
	ctx, done := harness.Phase(ctx, "sca")
	defer done()

	chunks := st.KB.Chunks
	view, viewID := SelectSCAView(chunks, ViewTerms(st))
	if st.SCAViewID != "" && viewID == st.SCAViewID {
		// Same evidence review twice despite new storage — further rounds cannot
		// change the verdict; stop instead of looping.
		logger.Printf("[SCA] review view UNCHANGED since last round (%d stored / %d viewed); closing out.", len(chunks), len(view))
		st.Verdict = VerdictInsufficient
		st.NoProgress = true
		st.SCA = map[string]any{}
		return
	}

	draftText := strings.TrimSpace(st.Draft)
	claims := buildSCAClaims(draftText, view, chunks, st.SlotEvidence)
	if len(claims) == 0 {
		logger.Printf("[SCA] nothing retrieved nor drafted; marking INSUFFICIENT to trigger a targeted re-search.")
		st.Verdict = VerdictInsufficient
		st.SCA = map[string]any{}
		st.SCAViewID = viewID
		return
	}

	// The SCA renders claim evidence from kbinfos BY INDEX, so review against a
	// view-only copy — the indexed ids then point at exactly the selected
	// chunks. The full pool is restored afterwards.
	orig := st.KB.Chunks
	st.KB.Chunks = view
	defer func() { st.KB.Chunks = orig }()

	t := min(SCATimeoutS, max(15.0, st.RemainingS()-10.0))
	callCtx, cancel := context.WithTimeout(ctx, time.Duration(t*float64(time.Second)))
	defer cancel()

	res := orchestrator.SufficientContextAgent(callCtx, orchestrator.SCADeps{
		KB:      st.KB,
		Model:   &jsonModelAdapter{inner: deps.Model},
		Prompts: deps.SCAPrompts,
	}, st.Question, claims)

	st.SCAViewID = viewID
	st.SCA = scaResultToMap(res)
	if len(st.SCA) == 0 {
		logger.Printf("[SCA] unavailable; accepting current draft.")
		st.Verdict = VerdictSufficient
		return
	}
	if res.IsSufficient {
		st.Verdict = VerdictSufficient
	} else {
		st.Verdict = VerdictInsufficient
	}
	logger.Printf("[SCA] verdict=%s (confidence=%.2f; view=%d/%d)", st.Verdict, res.Confidence, len(view), len(chunks))
}

// queryRewriteNode mirrors the `query_rewrite` node (line 1029): Phase-4
// targeted gap pursuit.
func queryRewriteNode(ctx context.Context, deps RAGTools, st *AgenticState, logger *log.Logger) {
	ctx, done := harness.Phase(ctx, "rewrite")
	defer done()

	gaps := SCAGapsToRewrite(st.SCA)
	if len(gaps) == 0 {
		logger.Printf("[QueryRewriter] SCA insufficient but no concrete gap; accepting the draft.")
		st.NoProgress = true
		return
	}

	// Information-augmented rewriting: give the rewriter FULL VISIBILITY — what
	// was tried (with outcomes), what the evidence pool holds — so it aims at
	// uncovered angles itself, instead of rule-based dedupe.
	researchContext := renderResearchContext(st)

	t := min(RewriteTimeoutS, max(10.0, st.RemainingS()-10.0))
	callCtx, cancel := context.WithTimeout(ctx, time.Duration(t*float64(time.Second)))
	defer cancel()

	rewritten := orchestrator.RewriteGapToQuery(callCtx, orchestrator.RewriteDeps{
		Model:           &jsonModelAdapter{inner: deps.Model},
		Prompts:         deps.RewritePrompts,
		ResearchContext: researchContext,
	}, st.Question, gaps)

	var queries []string
	for _, q := range rewritten {
		if s := strings.TrimSpace(q["query"]); s != "" {
			queries = append(queries, s)
		}
	}
	// Fold the still-unresolved slots into the rewrite queries so the next slot
	// research pass targets exactly those unknowns. This makes an insufficient
	// verdict drive slot completion, not a blind re-search.
	for _, us := range st.UnresolvedSlots {
		if clues, ok := us["question_clues"].([]string); ok {
			for i, qc := range clues {
				if i >= 2 {
					break
				}
				if s := strings.TrimSpace(qc); s != "" {
					queries = append(queries, s)
				}
			}
		}
	}
	queries = dedupe(queries)
	if len(queries) == 0 {
		logger.Printf("[QueryRewriter] no actionable query produced; accepting the draft.")
		st.NoProgress = true
		return
	}

	// Dual-track pursuit of the gap: (a) programmatically pre-fetch new snippets
	// into the SCA pool; (b) the next slot research pass picks these up via the
	// persisted slot_table + unresolved_slots.
	added := FanoutSearch(callCtx, deps, st, queries, FanoutTopNRewrite, MaxSnippetPool)
	// Retrieval saturation early-exit: a rewrite round that produced ZERO new
	// snippets means further full research passes just burn latency.
	if added == 0 && st.SearchRounds >= 1 {
		logger.Printf("[QueryRewriter] retrieval saturated (0 new chunks after another insufficient round); stopping iteration.")
		st.NoProgress = true
		st.CurrentQueries = queries
		return
	}

	logger.Printf("[QueryRewriter] insufficient round %d → %d targeted query(s): %v",
		st.SearchRounds+1, len(queries), queries)
	st.NoProgress = false
	st.CurrentQueries = queries
	st.SearchRounds++
	for _, q := range queries {
		st.Attempted = append(st.Attempted, map[string]any{"q": q, "r": st.SearchRounds, "new": added})
	}
}

// formalizeAnswerNode mirrors the `formalize_answer` node's state mutation
// (line 1118). The answer composition itself (Phase 5 synthesis) is out of
// scope here — it needs the report prompt templates; the caller reads the
// approved draft from st.KB.PreSummary.
func formalizeAnswerNode(st *AgenticState, logger *log.Logger) {
	if st.NoProgress || st.Verdict == VerdictInsufficient {
		// All research attempts exhausted without a satisfying context — surface
		// the residual findings honestly instead of refusing.
		st.PartialAnswer = true
	}
	st.EmptyResult = len(st.KB.Chunks) == 0
	logger.Printf("[Finalize] partial=%v empty=%v chunks=%d", st.PartialAnswer, st.EmptyResult, len(st.KB.Chunks))
}

// ---------------------------------------------------------------------------
// SCA helpers (Python agentic_rag_graph.py: _select_sca_view, _view_terms,
// _sca_gaps_to_rewrite, and the sca node's claim construction).
// ---------------------------------------------------------------------------

// buildSCAClaims mirrors the sca node's claim construction (lines 971-997): the
// draft as claim "c0", plus one claim per slot carrying its evidence positions.
func buildSCAClaims(draftText string, view, chunks []map[string]any, slotEvidence map[string]SlotEvidence) []orchestrator.ClaimDraft {
	var claims []orchestrator.ClaimDraft
	if draftText != "" {
		claims = append(claims, orchestrator.ClaimDraft{ID: "c0", Draft: draftText})
	}
	viewByID := map[string]bool{}
	for _, c := range view {
		if id := harness.ChunkIDOf(c); id != "" {
			viewByID[id] = true
		}
	}
	chunkByID := map[string]map[string]any{}
	for _, c := range chunks {
		if id := harness.ChunkIDOf(c); id != "" {
			chunkByID[id] = c
		}
	}
	sids := make([]string, 0, len(slotEvidence))
	for sid := range slotEvidence {
		sids = append(sids, sid)
	}
	sort.Strings(sids)
	for _, sid := range sids {
		meta := slotEvidence[sid]
		if len(meta.EvidenceIDs) == 0 {
			continue
		}
		// Resolve chunk-id -> view, appending missing chunks so the SCA sees the
		// passages that actually produced each candidate.
		for _, eid := range meta.EvidenceIDs {
			if viewByID[eid] {
				continue
			}
			if c, ok := chunkByID[eid]; ok {
				view = append(view, c)
				viewByID[eid] = true
			}
		}
		draft := truncateRunes(meta.Candidate, draftCandidateChars)
		if draft == "" {
			draft = fmt.Sprintf("(slot %s evidence)", sid)
		}
		claims = append(claims, orchestrator.ClaimDraft{ID: sid, Draft: draft, EvidenceIDs: meta.EvidenceIDs})
	}
	if len(claims) == 0 {
		claims = []orchestrator.ClaimDraft{{ID: "c0", Draft: draftText}}
	}
	return claims
}

// genJSONMaxRetry mirrors Python gen_json's default max_retry=2: the first
// call, plus one corrective round that feeds the malformed answer and the parse
// error back to the model.
const genJSONMaxRetry = 2

// jsonModelAdapter adapts harness.SessionModel to orchestrator.JSONModel.
//
// Mirrors Python's gen_json(prompt, "Output:\n", chat_mdl): the rendered prompt
// is the system turn and "Output:\n" the user turn, then the first JSON value is
// parsed out of the reply. Malformed JSON is retried (up to genJSONMaxRetry
// calls) with the model's own bad answer and the parse error appended to the
// user turn, so a single formatting hiccup does not abort the SCA review or the
// query rewrite. gen_json also consults an LLM reply cache and parses with
// json_repair's tolerant loader; Go has no equivalent cache, and the leniency is
// approximated by wrapper stripping plus a whole-object / brace-scan parse.
type jsonModelAdapter struct {
	inner harness.SessionModel
}

// genJSONTailFenceRE matches a trailing ``` fence followed by any newlines, the
// "```\n*$" alternative of gen_json's cleanup regex.
var genJSONTailFenceRE = regexp.MustCompile("```\\n*$")

// stripGenJSONWrappers mirrors gen_json's answer cleanup:
//
//	ans = re.sub(r"(^.*</think>|```json\n|```\n*$)", "", ans, flags=re.DOTALL)
//
// The think term (greedy up to the LAST </think>) is common.StripThinkTrailing;
// a "```json\n" fence may occur anywhere and is removed wholesale; a trailing
// "```" (plus newlines) is cut from the end.
func stripGenJSONWrappers(s string) string {
	s = common.StripThinkTrailing(s)
	s = strings.ReplaceAll(s, "```json\n", "")
	return genJSONTailFenceRE.ReplaceAllString(s, "")
}

// GenJSON implements orchestrator.JSONModel.
func (a *jsonModelAdapter) GenJSON(ctx context.Context, prompt string) (any, error) {
	if a.inner == nil {
		return nil, fmt.Errorf("agentic: no model configured")
	}
	var lastAns, errText string
	for attempt := 0; attempt < genJSONMaxRetry; attempt++ {
		userTurn := "Output:\n"
		if attempt > 0 && lastAns != "" && errText != "" {
			// gen_json appends the corrective prompt to the user turn only once
			// the previous round produced both an answer and a parse error.
			userTurn = fmt.Sprintf("Output:\n\nGenerated JSON is as following:\n%s\nBut exception while loading:\n%s\nPlease reconsider and correct it.", lastAns, errText)
		}
		reply, err := a.inner.Complete(ctx, []schema.Message{
			*schema.SystemMessage(prompt),
			*schema.UserMessage(userTurn),
		}, nil)
		if err != nil {
			// gen_json propagates chat/transport errors immediately; only JSON
			// parsing failures are retried. Mirror that.
			return nil, err
		}
		lastAns = reply.Content
		cleaned := stripGenJSONWrappers(reply.Content)
		var obj map[string]any
		if err := json.Unmarshal([]byte(strings.TrimSpace(cleaned)), &obj); err == nil {
			return obj, nil
		}
		if v := harness.ExtractJSON(cleaned); v != nil {
			return v, nil
		}
		// Deliberately log length, not content: the reply may embed retrieved
		// material (PII), so the raw body is excluded, as in jsonchat.GenJSON.
		errText = fmt.Sprintf("model reply (%d bytes) is not parseable JSON", len(cleaned))
	}
	return nil, fmt.Errorf("agentic: no parseable JSON in the model output after %d attempts", genJSONMaxRetry)
}

// scaResultToMap flattens an SCAResult back into the map shape the loop keeps in
// AgenticState.SCA (SCAGapsToRewrite reads both shapes).
func scaResultToMap(res orchestrator.SCAResult) map[string]any {
	if len(res.Claims) == 0 && res.Reasoning == "" && !res.IsSufficient {
		return nil
	}
	out := map[string]any{
		"is_sufficient":  res.IsSufficient,
		"confidence":     res.Confidence,
		"contradictions": toAnySlice(res.Contradictions),
		"reasoning":      res.Reasoning,
		"claims":         res.Claims,
	}
	if len(res.SubQueries) > 0 {
		sqs := make([]any, 0, len(res.SubQueries))
		for _, sq := range res.SubQueries {
			sqs = append(sqs, map[string]any{
				"sub_query":    sq.SubQuery,
				"satisfied":    sq.Satisfied,
				"missing_fact": sq.MissingFact,
				"search_hint":  sq.SearchHint,
			})
		}
		out["sub_queries"] = sqs
	}
	return out
}

func toAnySlice(ss []string) []any {
	out := make([]any, 0, len(ss))
	for _, s := range ss {
		out = append(out, s)
	}
	return out
}

// verdictStatusHint mirrors agentic_rag.py:903-905 — the human phrase folded
// into rag()'s "[Research status]" note for each sufficiency status.
func verdictStatusHint(verdict string) string {
	switch verdict {
	case VerdictInsufficient:
		return "still don't have enough evidence to answer"
	case VerdictSufficient:
		return "the question is answered"
	default:
		return "sufficiency status: " + verdict
	}
}

// scaFeedback mirrors agentic_rag.py:878 / :902-921 — the body rag() folds into
// the answer as the "[Research status]" note whenever research stays
// unsatisfying. Python builds it from sca.to_grounded(): a status hint,
// "hard_violations" (== Go's contradictions), "missing_claims", and
// "agent_confidence". Go renders the equivalent from the SCA verdict map
// produced by scaResultToMap. It is non-empty for any non-SUFFICIENT verdict, so
// the caller can always append the appropriate trailing sentence ("STOP" vs
// "call rag again") — matching Python, which emits the note for EVERY
// non-SUFFICIENT verdict, not only after two consecutive misses.
func scaFeedback(sca map[string]any, verdict string) string {
	if verdict == VerdictSufficient {
		return ""
	}
	var b strings.Builder
	b.WriteString(verdictStatusHint(verdict))
	// hard_violations == contradictions surfaced by the SCA (Python: hard_violations).
	if cons := scaContradictionStrings(sca); len(cons) > 0 {
		b.WriteString(" hard violations: " + strings.Join(cons, "; ") + ".")
	}
	// missing_claims — ungrounded claims and their missing_information (Python: missing_claims).
	if miss := scaMissingStrings(sca); len(miss) > 0 {
		b.WriteString(" missing claims: " + strings.Join(miss, "; ") + ".")
	}
	if conf, ok := sca["confidence"].(float64); ok {
		b.WriteString(fmt.Sprintf(" (confidence: %.2f)", conf))
	} else {
		b.WriteString(" (confidence: 0.00)")
	}
	return b.String()
}

// scaContradictionStrings extracts the SCA contradictions (Python's
// hard_violations) as a flat string slice.
func scaContradictionStrings(sca map[string]any) []string {
	var out []string
	if cons, ok := sca["contradictions"].([]any); ok {
		for _, c := range cons {
			if s, ok := c.(string); ok && s != "" {
				out = append(out, s)
			}
		}
	}
	return out
}

// scaMissingStrings extracts the SCA ungrounded-claim missing-information pieces
// (Python's missing_claims), capped the way scaFeedback caps them.
func scaMissingStrings(sca map[string]any) []string {
	var parts []string
	if claims, ok := sca["claims"].(map[string]any); ok {
		for cid, raw := range claims {
			entry, ok := raw.(map[string]any)
			if !ok {
				continue
			}
			// Python surfaces ungrounded claims and their missing_information.
			if grounded, _ := entry["grounded"].(bool); grounded {
				continue
			}
			if miss, ok := entry["missing_information"].([]any); ok && len(miss) > 0 {
				for _, m := range miss {
					if mp, ok := m.(map[string]any); ok {
						if w, _ := mp["what"].(string); w != "" {
							parts = append(parts, cid+": "+w)
							continue
						}
					}
					if s, ok := m.(string); ok && s != "" {
						parts = append(parts, cid+": "+s)
					}
				}
			} else if u, ok := entry["ungrounded"].([]any); ok && len(u) > 0 {
				parts = append(parts, cid+": ungrounded")
			}
		}
	}
	const maxFeedback = 4
	if len(parts) > maxFeedback {
		parts = parts[:maxFeedback]
	}
	return parts
}

// renderResearchContext mirrors the query_rewrite node's context build
// (lines 1047-1066): the attempted-query ledger with outcomes, plus the first
// lines of the evidence pool.
func renderResearchContext(st *AgenticState) string {
	var historyLines []string
	for _, e := range st.Attempted {
		if e == nil {
			continue
		}
		q := truncateRunes(stringOf(e["q"]), 120)
		r := stringOf(e["r"])
		if r == "" {
			r = "?"
		}
		outcome := "no new passages"
		if n, ok := toIntStrict(e["new"]); ok && n != 0 {
			outcome = fmt.Sprintf("%d new passage(s)", n)
		}
		historyLines = append(historyLines, fmt.Sprintf("- %s (round %s: %s)", q, r, outcome))
	}
	var poolLines []string
	if st.KB != nil {
		for i, c := range st.KB.Chunks {
			if i >= PoolHeadLines {
				break
			}
			first := harness.ChunkTextOf(c)
			if idx := strings.IndexByte(first, '\n'); idx >= 0 {
				first = first[:idx]
			}
			if first = strings.TrimSpace(first); first != "" {
				poolLines = append(poolLines, "- "+truncateRunes(first, 140))
			}
		}
	}
	var parts []string
	if len(historyLines) > 0 {
		parts = append(parts, "Previously searched queries and their outcomes:\n"+strings.Join(historyLines, "\n"))
	}
	if len(poolLines) > 0 {
		parts = append(parts, "Evidence currently at hand (first lines of top stored snippets):\n"+strings.Join(poolLines, "\n"))
	}
	return strings.Join(parts, "\n\n")
}

// routeSCA mirrors Python _route_sca (line 1130).
func routeSCA(st *AgenticState, enableSCA bool, scaMaxRounds int) agenticNode {
	if st.NoProgress {
		return nodeFormalizeAnswer
	}
	if !enableSCA {
		// medium: single research pass — the SCA verdict is informational only.
		return nodeFormalizeAnswer
	}
	if st.Verdict == VerdictInsufficient &&
		st.SearchRounds < scaMaxRounds &&
		// Budget guard: only start another rewrite+research round when enough
		// headroom remains for the whole round.
		st.RemainingS() > MinRoundHeadroomS {
		return nodeQueryRewrite
	}
	return nodeFormalizeAnswer
}

// routeRewrite mirrors Python _route_rewrite (line 1146).
func routeRewrite(st *AgenticState, scaMaxRounds int, logger *log.Logger) agenticNode {
	if st.NoProgress {
		return nodeFormalizeAnswer
	}
	if st.SearchRounds >= scaMaxRounds {
		return nodeFormalizeAnswer
	}
	if st.RemainingS() <= MinRoundHeadroomS {
		if logger != nil {
			logger.Printf("[Routing] research budget nearly exhausted (%.0fs left); closing out with current evidence.", st.RemainingS())
		}
		return nodeFormalizeAnswer
	}
	return nodeRagAgentLoop
}

// RenderSlotDraft mirrors Python _render_slot_draft: render the slot table into
// a fact-preserving draft for the SCA.
func RenderSlotDraft(slotTable harness.State, collectedAnswer string, slotEvidence map[string]SlotEvidence) string {
	var b strings.Builder
	if len(slotTable.State) == 0 {
		if collectedAnswer != "" {
			return collectedAnswer
		}
		return ""
	}
	for _, v := range slotTable.State {
		b.WriteString(fmt.Sprintf("- slot %d (%s): ", v.ID, v.Type))
		if v.Candidate != nil && *v.Candidate != "" {
			b.WriteString(truncateRunes(*v.Candidate, draftCandidateChars))
		} else {
			b.WriteString("UNRESOLVED")
		}
		b.WriteString("\n")
		if len(v.QuestionClues) > 0 {
			b.WriteString("  asked: " + strings.Join(truncateEach(v.QuestionClues, draftClueChars), "; ") + "\n")
		}
		if ev, ok := slotEvidence[fmt.Sprint(v.ID)]; ok && len(ev.EvidenceIDs) > 0 {
			b.WriteString(fmt.Sprintf("  evidence: %d passage(s)\n", len(ev.EvidenceIDs)))
		}
	}
	if collectedAnswer != "" {
		b.WriteString("\nCandidate answer:\n" + collectedAnswer + "\n")
	}
	return b.String()
}

// ---------------------------------------------------------------------------
// Slot-table research: the executor behind the outer loop's `rag_agent` node.
//
// Mirrors Python agentic_rag_graph.py:
//   - _build_slot_table        (line 1243)
//   - _run_slot_research_pass  (line 1278)
//   - _merge_slot_patch        (line 1409)
//   - _render_slot_draft       (line 1191)
//
// One research round = one action session per unresolved slot, run concurrently
// under a semaphore, with the resulting branches folded back into the shared
// table.
// ---------------------------------------------------------------------------

// SlotEvidence records which passages produced a slot's candidate, so the SCA
// can verify the candidate against the passages that actually produced it.
type SlotEvidence struct {
	EvidenceIDs  []string
	TerminalType string
	Candidate    string
	Strength     *float64
}

// SlotResearchResult is one research round's output.
type SlotResearchResult struct {
	SlotTable       harness.State
	CollectedAnswer string
	UnresolvedSlots []map[string]any
	SlotEvidence    map[string]SlotEvidence
	SlotDraft       string
	Attempted       []map[string]any
}

// BuildSlotTable mirrors Python _build_slot_table: decompose the question into
// a slot table, seeding it with the planner's fan-outs.
//
// Returns (root, firstQueries) — never an empty root: on failure it degrades to
// one "aspect" slot per fan-out (or a single "answer" slot for the raw
// question), so the research pass always has something to work on.
func BuildSlotTable(ctx context.Context, deps harness.SessionDeps, question string, fanouts []string, deadlineLeft float64) (harness.State, []string) {
	var root harness.State
	firstQueries, err := buildSlotTableFrom(ctx, deps, question, fanouts, deadlineLeft, &root)
	if err != nil {
		_LOG.Printf("[SlotTable] initialize_state failed; building from fanouts: %v", err)
	}
	if len(root.State) == 0 {
		queries := fanouts
		if len(queries) == 0 {
			queries = []string{question}
		}
		vars := make([]harness.Variable, 0, 4)
		for i, q := range queries {
			if i >= 4 {
				break
			}
			vars = append(vars, harness.Variable{
				ID:            i,
				Type:          "aspect",
				QuestionClues: []string{truncateRunes(q, slotFallbackClueChars)},
			})
		}
		root = harness.NewState(vars, 0, nil)
		if len(firstQueries) == 0 {
			firstQueries = queries
			if len(firstQueries) > 3 {
				firstQueries = firstQueries[:3]
			}
		}
	}
	_LOG.Printf("[SlotTable] built %d slot(s): %s", len(root.State), root.Brief())
	if len(firstQueries) == 0 {
		firstQueries = []string{question}
	}
	return root, firstQueries
}

// buildSlotTableFrom wraps InitializeState, converting its non-error result into
// an error so BuildSlotTable can apply the single fallback path.
func buildSlotTableFrom(ctx context.Context, deps harness.SessionDeps, question string, fanouts []string, deadlineLeft float64, out *harness.State) ([]string, error) {
	if deps.Model == nil {
		return nil, fmt.Errorf("no model configured")
	}
	init := harness.InitializeState(ctx, deps, question, fanouts, deadlineLeft)
	if len(init.Root.State) == 0 {
		return nil, fmt.Errorf("decomposition produced no slots")
	}
	*out = init.Root
	return init.FirstQueries, nil
}

// RunSlotResearchPass mirrors Python _run_slot_research_pass: drive ONE
// research round with slot-aware action sessions.
//
// Unresolved slots are worked concurrently under a semaphore; each session's
// branches are folded back into the shared table. A nil result means "nothing to
// do" (all slots already filled).
func RunSlotResearchPass(ctx context.Context, deps harness.SessionDeps, question string, st *AgenticState, deadlineLeft float64) *SlotResearchResult {
	slotTable := st.SlotTable
	if len(slotTable.State) == 0 {
		// No planner ran (medium single-pass, or the planner failed): build the
		// table from the raw question so the research still executes.
		root, _ := BuildSlotTable(ctx, deps, question, nil, max(15.0, deadlineLeft-10.0))
		slotTable = root
	}
	if question == "" {
		question = st.Question
	}
	unresolved := slotTable.Unresolved()
	if len(unresolved) == 0 {
		_LOG.Printf("[SlotResearch] all slots filled; no session to run.")
		return nil
	}

	// Shared across sessions so duplicate retrievals are served from cache.
	sharedToolCache := map[string]harness.ToolOutcome{}
	var sharedSearchQueries []string

	sem := make(chan struct{}, slotSessionConcurrency)
	var wg sync.WaitGroup
	var mu sync.Mutex

	type outcome struct {
		slotID int
		result harness.Result
	}
	results := make([]outcome, 0, slotSessionsPerRound)
	limit := min(len(unresolved), slotSessionsPerRound)
	sessionBudget := max(20.0, deadlineLeft-10.0)

	for i := 0; i < limit; i++ {
		v := unresolved[i]
		wg.Add(1)
		go func(v harness.Variable) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			direction := question
			if len(v.QuestionClues) > 0 {
				direction = v.QuestionClues[0]
			}
			// Sessions share ONE Toolset; DisableTool mutates it, so the call is
			// guarded here. The Kbinfos merge happens inside the executor.
			res := harness.RunActionSession(ctx, deps, direction, slotTable, sessionBudget, "", sharedToolCache, sharedSearchQueries)
			mu.Lock()
			results = append(results, outcome{slotID: v.ID, result: res})
			mu.Unlock()
		}(v)
	}
	wg.Wait()

	// Fold in slot-id order so the merge is deterministic regardless of which
	// session finished first.
	sort.Slice(results, func(i, j int) bool { return results[i].slotID < results[j].slotID })

	collected := st.CollectedAnswer
	sessionEvidence := map[string]SlotEvidence{}
	ledger := append([]map[string]any(nil), st.Attempted...)

	for _, item := range results {
		r := item.result
		if r.FoundAnswer != nil && collected == "" {
			collected = *r.FoundAnswer
		}
		if len(r.RetrievedEvidenceIDs) > 0 {
			terminalType := ""
			if r.TerminalType != nil {
				terminalType = *r.TerminalType
			}
			var candidate string
			if r.FoundAnswer != nil {
				candidate = *r.FoundAnswer
			}
			sessionEvidence[fmt.Sprint(item.slotID)] = SlotEvidence{
				EvidenceIDs:  dedupe(r.RetrievedEvidenceIDs),
				TerminalType: terminalType,
				Candidate:    candidate,
			}
		}
		for _, ns := range r.NewStates {
			if merged := MergeSlotPatch(slotTable, ns); merged != nil {
				slotTable = *merged
			}
		}
		q := ""
		if r.FoundAnswer != nil {
			q = truncateRunes(*r.FoundAnswer, 80)
		}
		ledger = append(ledger, map[string]any{"q": q, "new": 1})
	}

	unresolvedOut := make([]map[string]any, 0, len(unresolved))
	for _, v := range slotTable.Unresolved() {
		clues := v.DiscoveredClues
		if len(clues) > 4 {
			clues = clues[len(clues)-4:]
		}
		unresolvedOut = append(unresolvedOut, map[string]any{
			"id":               v.ID,
			"type":             v.Type,
			"question_clues":   append([]string(nil), v.QuestionClues...),
			"discovered_clues": append([]string(nil), clues...),
		})
	}

	draft := RenderSlotDraft(slotTable, collected, sessionEvidence)
	_LOG.Printf("[SlotResearch] round done — %d slot(s) filled, unresolved=%d, collected_answer=%v",
		countFilled(slotTable), len(unresolvedOut), collected != "")
	_LOG.Printf("[SlotResearch] slot table after round:\n%s", draft)

	return &SlotResearchResult{
		SlotTable:       slotTable,
		CollectedAnswer: collected,
		UnresolvedSlots: unresolvedOut,
		SlotEvidence:    sessionEvidence,
		SlotDraft:       draft,
		Attempted:       ledger,
	}
}

// MergeSlotPatch mirrors Python _merge_slot_patch: fold a session's new-state
// branch into the shared slot table, adopting the STRONGER candidate.
//
// Returns nil when nothing changed (mirrors Python's `if not changed: return
// None`), so callers can skip no-op merges.
func MergeSlotPatch(base, branch harness.State) *harness.State {
	if len(branch.State) == 0 {
		return nil
	}
	branchByID := map[int]harness.Variable{}
	for _, v := range branch.State {
		branchByID[v.ID] = v
	}
	merged := make([]harness.Variable, 0, len(base.State))
	changed := false
	for _, v := range base.State {
		bv, ok := branchByID[v.ID]
		if !ok {
			merged = append(merged, v)
			continue
		}
		// Adopt the branch candidate only when STRONGER. Sessions run
		// concurrently and their branches fold in completion order, so an
		// unconditional "branch wins" made the result both order-dependent and
		// destructive: a weak session (0.3, tentative) could downgrade a slot
		// another session had already proven (0.95).
		cand, strength := v.Candidate, v.CandidateStrength
		switch {
		case bv.Candidate == nil:
			// branch has no candidate: keep base
		case v.Candidate == nil:
			cand, strength = bv.Candidate, bv.CandidateStrength
		case strengthOf(bv) > strengthOf(v):
			cand, strength = bv.Candidate, bv.CandidateStrength
		}
		clues := dedupe(append(append([]string(nil), v.DiscoveredClues...), bv.DiscoveredClues...))
		if !equalStringPtr(cand, v.Candidate) || !equalStrings(clues, v.DiscoveredClues) {
			changed = true
		}
		merged = append(merged, harness.Variable{
			ID:                v.ID,
			Type:              v.Type,
			QuestionClues:     append([]string(nil), v.QuestionClues...),
			DiscoveredClues:   clues,
			Candidate:         cand,
			CandidateStrength: strength,
		})
	}
	if !changed {
		return nil
	}
	out := harness.NewState(merged, base.Depth+1, append([]string(nil), base.RetrievedEvidenceIDs...))
	return &out
}

func strengthOf(v harness.Variable) float64 {
	if v.CandidateStrength == nil {
		return 0.0
	}
	return *v.CandidateStrength
}

// ComposeFallbackDraft mirrors Python _compose_fallback_draft (line 1466): an
// intermediate draft synthesized from the snippet pool when the research pass
// produced no report (budget exhaustion).
//
// The Go port is a deterministic concatenation of the top snippets rather than
// an LLM call — no extra model round-trip, and it preserves facts verbatim.
func ComposeFallbackDraft(ctx context.Context, deps RAGTools, st *AgenticState) string {
	// Nil-safe: the draft node runs on every round, including a state whose KB
	// was never populated (Python reaches the same fields through getattr, which
	// tolerates a missing pool).
	if st == nil || st.KB == nil || len(st.KB.Chunks) == 0 {
		return ""
	}
	var b strings.Builder
	b.WriteString(fmt.Sprintf("Question: %s\n\nFindings so far:\n", st.Question))
	n := min(len(st.KB.Chunks), 10)
	for i := 0; i < n; i++ {
		if ctx.Err() != nil {
			break
		}
		if txt := strings.TrimSpace(harness.ChunkTextOf(st.KB.Chunks[i])); txt != "" {
			b.WriteString("- " + truncateRunes(txt, 400) + "\n")
		}
	}
	return b.String()
}

// NaiveRAG mirrors Python _naive_rag (line 1517): answer with one retrieve pass
// and no agentic graph at all.
//
// Used when the thinking mode is unrecognised — instead of failing the request
// (the label comes from user input) this degrades to plain retrieval plus one
// composed answer.
//
// Python is an async generator that yields the answer; Go returns it whole, so
// this mirrors the contract, not the streaming shape.
//
// The composition itself lives in ComposeNaiveAnswer, which mirrors
// Python lines 1555-1568 (fixed short prompt, flat "[i] content" evidence over
// the first 8 chunks at 1500 chars, message_fit_in, temperature 0.3, and the
// raw evidence as the failure fallback).
func NaiveRAG(ctx context.Context, deps RAGTools, req harness.RunRequest) *RunResponse {
	logger := deps.Logger
	if logger == nil {
		logger = _LOG
	}
	question := strings.TrimSpace(req.Question)
	resp := &RunResponse{EmptyResult: true, Mode: harness.GetMode("naive")}

	logger.Printf("[Naive RAG] single-pass retrieval for question_len=%d", len(question))

	// Python 1533: tools.retrieve(question) — one pass, no keywords, and a
	// retrieval failure degrades to "no evidence" rather than erroring.
	kb := &harness.Kbinfos{}
	if question != "" {
		chunks, aggs := harness.HybridSearch(ctx, harness.SearchDeps{
			Backend:        deps.Retriever,
			KbIDs:          req.DatasetIDs,
			TenantID:       req.TenantID,
			KB:             kb,
			UsingEmbedding: deps.UsingEmbedding,
		}, harness.SearchParams{
			Question: question,
			TopN:     req.TopN,
		})
		kb.Merge(chunks, aggs)
	}

	resp.Chunks = kb.Chunks
	resp.DocAggs = kb.DocAggs
	resp.Kbinfos = kb
	if len(kb.Chunks) == 0 {
		// Python 1539-1541: yield the configured empty response.
		resp.Answer = deps.EmptyResponse
		return resp
	}
	resp.EmptyResult = false

	out := ComposeNaiveAnswer(ctx, AnswerDeps{
		Model:         deps.Model,
		EmptyResponse: deps.EmptyResponse,
		MaxLength:     deps.MaxLength,
		Logger:        logger,
	}, kb.Chunks, question)

	resp.Answer = out.Answer
	if out.NoEvidence || out.Failed {
		resp.EmptyResult = out.NoEvidence
	}
	return resp
}

// countFilled counts slots holding a candidate.
func countFilled(slotTable harness.State) int {
	n := 0
	for _, v := range slotTable.State {
		if v.Filled() {
			n++
		}
	}
	return n
}

// ---------------------------------------------------------------------------
// Graph driver — mirrors Python build_agentic_graph (line 773).
// ---------------------------------------------------------------------------

// BuildAgenticGraph drives the agentic-search loop and returns the terminal
// state.
//
// Mirrors Python build_agentic_graph (line 773): Python declares the nodes with
// StateGraph and compile()s them, then run_agentic_rag invokes the result
// (:1614). Go has no graph runtime and therefore no compile step — this runs the
// nodes directly, so it both "builds" and "runs" the graph. The name follows
// Python's build_agentic_graph for traceability.
//
// The caller reads the answer from st.KB.PreSummary / st.CollectedAnswer and
// composes it (see Run for the single-shot entry point).
//
// Activation is EXPLICIT: call this directly (corresponding to Python's
// dialog_service.py instantiating RAGTools and invoking agentic_rag). There is
// no init()-based auto-registration into the harness.
//
// This is the agentic planner (medium/high/ultra). The mode dispatch lives in the
// package-local RAGTools.Run; agentic modes register this as the AgenticLoop.
func BuildAgenticGraph(ctx context.Context, deps RAGTools, question, keywords string, maxLoops int, messages []schema.Message) *AgenticState {
	logger := deps.logger()
	// Python :761-768 — the graph starts at formalize_question, which both
	// initializes the state and arms the budget, so the budget is NOT set here.
	st := NewAgenticState(question, keywords, maxLoops, messages)
	if deps.KB != nil {
		st.KB = deps.KB
	}
	spec := harness.ResolveMode(deps.Tools)
	enableSCA := spec.EnableSCA
	useFanout := spec.UseFanout
	scaMaxRounds := spec.SCAMaxRounds

	logger.Printf("[Agentic RAG] Starting research — mode=%s sca=%v fanouts=%v",
		spec.Label, enableSCA, useFanout)

	// Hard ceiling: the routing guards steer to synthesis before this fires; it
	// only trips if a node still hangs so the caller never waits forever.
	runCtx, cancel := context.WithTimeout(ctx, time.Duration(TotalBudgetS+30.0)*time.Second)
	defer cancel()

	// Prefetch is gated on fan-out, mirroring Python's `use_prefetch =
	// use_fanout`. NOTE: the Python comment there claims prefetch is DISABLED,
	// but the code still wires it for fan-out modes — behaviour here matches the
	// CODE, not the stale comment.
	usePrefetch := useFanout

	// START → formalize_question (Python :765). The node itself routes onward
	// the same way Python's edges do.
	node := nodeFormalizeQuestion
	firstRound := true
	visits := 0

	// Python :1619 — LangGraph counts NODE VISITS, not loop iterations: one
	// research round is rag_agent → draft → sca = three visits. Counting
	// iterations instead would let Go do roughly 3x the work before the guard
	// trips, so each branch adds the number of nodes it actually executes.
	limit := graphRecursionLimit(true, maxLoops)

	for {
		if visits >= limit {
			if deps.Logger != nil {
				deps.Logger.Printf("[Agentic RAG] stopping after %d node visits (Python recursion_limit=%d)", visits, limit)
			}
			return st
		}
		switch node {
		case nodeFormalizeQuestion:
			visits++
			formalizeQuestionNode(runCtx, deps, st, logger)
			// Python :1180-1183 — onward to the planner, or straight to
			// rag_agent when fan-out is off (medium).
			if useFanout {
				node = nodePlanner
			} else {
				node = nodeRagAgentFirst
			}

		case nodePlanner:
			visits++
			plannerNode(runCtx, deps, st, logger)
			if usePrefetch {
				node = nodePrefetch
			} else {
				node = nodeRagAgentFirst
			}

		case nodePrefetch:
			visits++
			prefetchNode(runCtx, deps, st, logger, firstRound)
			node = nodeRagAgentFirst

		case nodeRagAgentFirst, nodeRagAgentLoop:
			visits += agenticRoundVisits // rag_agent + draft + sca
			ragAgentNode(runCtx, deps, st, logger)
			draftNode(runCtx, deps, st, logger)
			scaNode(runCtx, deps, st, logger)
			node = routeSCA(st, enableSCA, scaMaxRounds)
			firstRound = false

		case nodeQueryRewrite:
			visits++
			queryRewriteNode(runCtx, deps, st, logger)
			node = routeRewrite(st, scaMaxRounds, logger)

		case nodeFormalizeAnswer:
			formalizeAnswerNode(st, logger)
			return st

		default:
			return st
		}
	}
}

// ---------------------------------------------------------------------------
// Explicit wiring into the RAGTools.Run loop registration (mirrors Python
// dialog_service.py instantiating RAGTools and driving the agentic graph).
//
// The Python module has NO init()/auto-registration: the caller constructs the
// object and invokes it. Go mirrors that — callers activate the full
// medium/high/ultra pipeline by explicitly registering this loop:
//
//	advanced_rag.SetAgenticLoop(advanced_rag.NewAgenticLoop())
//
// Without registration, RAGTools.Run falls back to a single action session for
// agentic modes (see Run).
// ---------------------------------------------------------------------------

// NewAgenticLoop returns the outer agentic loop adapter for RAGTools.Run. It
// converts the RAGTools run config into the planner's state and maps the result
// back onto the
// RunResponse.
func NewAgenticLoop() AgenticLoop {
	return func(ctx context.Context, deps RAGTools, req harness.RunRequest, kb *harness.Kbinfos, resp *RunResponse, logger *log.Logger) {
		if logger == nil {
			logger = _LOG
		}
		if deps.Model == nil {
			logger.Printf("[Agentic RAG] no model configured for mode %q; degrading to a direct search", resp.Mode.Label)
			runDirectFallback(ctx, deps, req, kb, logger)
			return
		}

		// The loop needs an executor for its programmatic fan-out fetches, built
		// from the same retrieval backend the single-session path uses.
		sd := harness.SearchDeps{
			Backend:        deps.Retriever,
			KbIDs:          req.DatasetIDs,
			TenantID:       req.TenantID,
			KB:             kb,
			Model:          deps.Model,
			WebSearch:      deps.WebSearch,
			UsingEmbedding: deps.UsingEmbedding,
			CiteRules:      deps.CiteRules,
		}
		if sd.Backend == nil {
			sd.Backend = &harness.RuntimeRetriever{}
		}

		toolset := &harness.Toolset{
			ThinkingMode:  resp.Mode.Label,
			HasWebSearch:  resp.Mode.HasTool("web_search"),
			DisabledTools: map[string]bool{},
			Exec:          harness.NewSearchExecutor(sd, req),
		}

		st := BuildAgenticGraph(ctx, RAGTools{
			Tools:          toolset,
			Model:          deps.Model,
			Prompts:        deps.Prompts,
			KB:             kb,
			SCAPrompts:     deps.Prompts,
			RewritePrompts: deps.Prompts,
			MaxLength:      deps.MaxLength,
			Logger:         logger,
		}, req.Question, req.Keywords, 3, deps.Messages)

		// Surface the loop's outcome on the RunResponse.
		resp.Slots = append(resp.Slots, st.SlotTable.State...)
		// Research findings: the SCA-reviewed draft. NOTE: this is the research
		// draft, NOT the final answer — RAGTools.Run composes the final cited
		// answer afterwards from KB.PreSummary (see composeFinalAnswer), exactly
		// as Python's formalize_answer node does.
		if st.CollectedAnswer != "" {
			resp.CollectedAnswer = st.CollectedAnswer
		}
		resp.Partial = st.PartialAnswer
		resp.SearchRounds = st.SearchRounds
		resp.Verdict = st.Verdict
		// SCAFeedback mirrors agentic_rag.py:878 / :902-921 — the body of the
		// "[Research status]" note (status hint + hard violations + missing
		// claims + confidence) that rag() folds into the answer for EVERY
		// non-SUFFICIENT verdict. Rag() appends the trailing "STOP" vs
		// "call rag again" sentence based on the consecutive-unanswerable count.
		resp.SCAFeedback = scaFeedback(st.SCA, st.Verdict)
		// Update the consecutive-unanswerable guardrail (Python
		// RAGTools._consecutive_unanswerable, :818). The counter lives on the
		// conversation-scoped *RAGCache so it survives across turns.
		if deps.Cache != nil {
			if st.Verdict == VerdictSufficient {
				deps.Cache.ConsecutiveUnanswerable = 0
			} else {
				deps.Cache.ConsecutiveUnanswerable++
			}
		}
	}
}

// RunAgenticRAG mirrors Python run_agentic_rag (agentic_rag_graph.py:1571):
// the mode dispatch plus execution of the selected pipeline. It is the
// counterpart of RAGTools.rag (Go: Run), which owns the conversation-level
// concerns around it — cache lookup, the effective question, attachments and
// storing the result.
//
// The dispatch is agentic_rag_graph.py:1581:
//
//	mode is NAIVE      → _naive_rag            (plain retrieval)
//	mode.Agentic       → build_agentic_graph   (medium / high / ultra)
//	else               → build_low_graph       (low: formalize → direct_search)
//
// Formalization belongs to the graphs, not to Run: Python makes it the first
// node of build_agentic_graph (:803) and build_low_graph (:738), while
// _naive_rag (:1517) has none. Each path below therefore runs it where its
// Python counterpart does.
// formalizeQuestion is Python's formalize_question node
// (build_agentic_graph:803, build_low_graph:738): resolve pronouns and ellipses
// from the conversation into a standalone question plus search keywords.
//
// Only the agentic and low graphs have this node; _naive_rag (:1517) has none,
// so callers decide whether to invoke it. Single-turn input costs no LLM call —
// Formalize returns early — mirroring Python's non-LLM fast path.
//
// The graph budget is NOT started here: Python sets that deadline in the node's
// return (:823), i.e. after formalization, so this cost is not charged to it.
// graphRecursionLimit mirrors Python run_agentic_rag:1619 — the graph aborts
// after this many node visits: 60 for the agentic graph, else
// max(25, max_loops*8). Go has no graph runtime, so the loop counts its own
// iterations against it.
func graphRecursionLimit(agentic bool, maxLoops int) int {
	if agentic {
		return AgenticRecursionLimit
	}
	if n := maxLoops * 8; n > lowRecursionLimitBase {
		return n
	}
	return lowRecursionLimitBase
}

// formalizeQuestionNode is Python's formalize_question node of
// build_agentic_graph (:803). It resolves pronouns and ellipses from the
// conversation into a standalone question plus search keywords, and arms the
// global budget in its return (:823) — so the formalization work itself is NOT
// charged to that budget.
//
// Single-turn input costs no LLM call (Formalize returns early), mirroring
// Python's non-LLM fast path.
func formalizeQuestionNode(ctx context.Context, deps RAGTools, st *AgenticState, logger *log.Logger) {
	// Python :823 — the budget is armed as the node returns, i.e. after any
	// formalization work has already happened.
	st.Deadline = time.Now().Add(time.Duration(TotalBudgetS * float64(time.Second)))

	if len(st.Messages) == 0 || deps.Model == nil {
		return
	}
	fdeps := harness.SessionDeps{Model: deps.Model, Prompts: deps.Prompts}
	q, kw := Formalize(ctx, fdeps, st.Messages, deps.MaxLength)
	if q != "" {
		st.Question = q
	}
	if kw != "" && st.Keywords == "" {
		st.Keywords = kw
	}
	if logger != nil && q != "" {
		logger.Printf("[Agentic RAG] formalized the question into %q", trunc(q, 80))
	}
}

// formalizeQuestion is Python's formalize_question node of build_low_graph
// (:738). The low graph has no scheduler in Go, so it runs as a plain step of
// BuildLowGraph rather than as a scheduled node.
func formalizeQuestion(ctx context.Context, deps RAGTools, req *harness.RunRequest, logger *log.Logger) {
	if len(deps.Messages) == 0 || deps.Model == nil {
		return
	}
	fdeps := harness.SessionDeps{Model: deps.Model, Prompts: deps.Prompts}
	if deps.Stats != nil {
		deps.Stats.RecordCall("formalize")
	}
	q, kw := Formalize(ctx, fdeps, deps.Messages, deps.MaxLength)
	if deps.Stats != nil && q == "" {
		deps.Stats.RecordFailed("formalize")
	}
	if q != "" {
		req.Question = q
	}
	if kw != "" && req.Keywords == "" {
		req.Keywords = kw
	}
	if logger != nil && q != "" {
		logger.Printf("[Agentic RAG] formalized the question into %q", trunc(q, 80))
	}
}

func RunAgenticRAG(ctx context.Context, deps RAGTools, req harness.RunRequest, sd harness.SearchDeps, kb *harness.Kbinfos, resp *RunResponse, logger *log.Logger, spec harness.ModeSpec) {
	// The graph budget (Python :823, mirrored by NewAgenticState) is NOT started
	// here: Python sets that deadline inside the first node's return, i.e. after
	// formalization has run, so formalization is not charged to it.
	switch {
	case spec.Agentic:
		// build_agentic_graph (:773) — formalization is the graph's first node,
		// so it happens inside runAgentic.
		runAgentic(ctx, deps, req, sd, kb, resp, logger)
	case spec.Label != "naive":
		// build_low_graph (:733) — formalize → direct_search.
		BuildLowGraph(ctx, deps, req, sd, kb, resp, logger)
	default:
		// _naive_rag (:1517) has no formalize node at all.
		runDirect(ctx, deps, req, sd, kb, resp, logger)
	}

	// Python :1667-1668 — only when research produced NOTHING *and* the graph
	// itself failed. An empty result without a failure is not an error: that is
	// EmptyResponse's job, and overwriting it here would hide the real reason.
	if resp.GraphFailed && resp.Answer == "" && len(resp.Slots) == 0 {
		logger.Printf("[Agentic RAG] research failed without producing anything")
		resp.Answer = graphFailureFallback
	}
}

// runDirectFallback retrieves once when no model is configured: the loop cannot
// plan, research, or review without one, so the caller still gets evidence
// rather than an error.
func runDirectFallback(ctx context.Context, deps RAGTools, req harness.RunRequest, kb *harness.Kbinfos, logger *log.Logger) {
	backend := deps.Retriever
	if backend == nil {
		backend = &harness.RuntimeRetriever{}
	}
	chunks, aggs := harness.HybridSearch(ctx, harness.SearchDeps{
		Backend:        backend,
		KbIDs:          req.DatasetIDs,
		TenantID:       req.TenantID,
		KB:             kb,
		UsingEmbedding: deps.UsingEmbedding,
	}, harness.SearchParams{
		Question:    req.Question,
		Keywords:    req.Keywords,
		UseCompiled: req.UseCompiled,
		TopN:        req.TopN,
	})
	kb.Merge(chunks, aggs)
}

// Final answer composition (the graph's last node).
//
// Mirrors Python agentic_rag_graph.py:
//   - _compose_answer_from_evidence (line 604)
//   - _naive_rag                   (line 1517)
//
// Every thinking mode ends here: it turns the gathered evidence into a grounded,
// cited answer in the user's language. The evidence-block and citation-rule
// prompts it renders (generator.py kb_prompt/citation_prompt) live in
// internal/rag/prompts; the system-prompt texts live in the harness package
// (report_prompt.go, mirroring report_prompt.py).

const (
	// evidenceBudgetTokens is the token ceiling of the evidence block
	// (Python agentic_rag._EVIDENCE_BUDGET_TOKENS).
	evidenceBudgetTokens = 8000
	// citeChunkCap caps chunks rendered as citation reference
	// (Python _CITE_CHUNK_CAP).
	citeChunkCap = 6
	// answerTimeoutS bounds the answer-composition call.
	answerTimeoutS = 150.0
	// naiveEvidenceChunkCap caps evidence chunks in the naive path
	// (Python: chunks[:8]).
	naiveEvidenceChunkCap = 8
	// naiveEvidenceCharCap caps each naive evidence chunk
	// (Python: [:1500]).
	naiveEvidenceCharCap = 1500
	// answerErrorFallback mirrors Python's stream-failure message.
	answerErrorFallback = "I'm sorry, I encountered an error while composing the answer."
	// graphFailureFallback mirrors Python run_agentic_rag's last-resort message
	// (:1668), used only when the graph failed AND produced nothing.
	graphFailureFallback = "I couldn't complete the search due to an internal error."
	// AgenticRecursionLimit mirrors Python run_agentic_rag:1619 for the agentic
	// graph: the maximum number of node visits before the graph aborts. Go has
	// no graph runtime, so the loop counts its own node visits against it.
	AgenticRecursionLimit = 60
	// agenticRoundVisits is the number of LangGraph node visits one research
	// round costs: rag_agent → draft → sca (Python :1181-1183).
	agenticRoundVisits = 3
	// lowRecursionLimitBase is the floor of Python's `max(25, max_loops * 8)`
	// (:1619) for the non-agentic graph.
	lowRecursionLimitBase = 25
)

// FinalAnswerSystem and PartialAnswerPreamble are defined in the harness
// package (harness/report_prompt.go), mirroring
// rag/advanced_rag/harness/prompts/report_prompt.py. They are re-used here via
// the harness import rather than duplicated.

var reAnswerThink = regexp.MustCompile(`(?s)^.*</think>`)

// AnswerDeps are the dependencies of ComposeAnswer.
type AnswerDeps struct {
	// Model drives the composition call.
	Model harness.SessionModel
	// CiteRules overrides DEFAULT_CITE_RULES. Empty uses the default.
	CiteRules string
	// SystemPrompt is the dialog-level UI configuration (Python
	// tools.system_prompt). Appended AFTER the agentic contract — see the
	// precedence note in composeSystem.
	SystemPrompt string
	// EmptyResponse is returned verbatim when there is no evidence
	// (Python tools.empty_response), skipping the LLM entirely.
	EmptyResponse string
	// MaxTokens caps the evidence block. <=0 uses evidenceBudgetTokens.
	MaxTokens int
	// MaxLength is the chat model's context window (Python
	// tools.chat_mdl.max_length). It bounds message_fit_in; <=0 falls back to
	// chat.EffectiveContextLength's 8192 default.
	MaxLength int
	// Logger is optional; nil uses the default logger.
	Logger *log.Logger
}

// AnswerResult is the composed final answer.
type AnswerResult struct {
	// Answer is the composed text.
	Answer string
	// Partial is true when the underlying research ended without a satisfying
	// verdict; the answer carries a partial-information preamble.
	Partial bool
	// NoEvidence is true when composition was skipped for lack of evidence.
	NoEvidence bool
	// Failed is true when the composition call failed and the fallback message
	// was returned.
	Failed bool
}

// ComposeAnswer mirrors Python _compose_answer_from_evidence: turn the gathered
// evidence into a grounded, cited answer.
//
// Behaviour, in Python's order:
//  1. no evidence + configured empty_response → return it WITHOUT calling the LLM;
//  2. rank chunks by similarity, keep the top citeChunkCap as citation reference;
//  3. render the evidence block under the token budget (kb_prompt);
//  4. prepend the fact-preserving pre_summary (the SCA-reviewed draft) when set;
//  5. call the model with FINAL_ANSWER_SYSTEM + the composed user content.
func ComposeAnswer(ctx context.Context, deps AnswerDeps, kb *harness.Kbinfos, question string, partial, abstain bool) AnswerResult {
	logger := deps.Logger
	if logger == nil {
		logger = _LOG
	}
	chunks := []map[string]any{}
	if kb != nil {
		chunks = kb.Chunks
	}
	note := ""
	if partial {
		note = " — partial answer, some gaps remain"
	} else if abstain {
		note = " — not enough evidence to answer"
	}
	logger.Printf("[Composing the answer] Writing the final answer to %q from %d gathered passage(s)%s.",
		trunc(question, 60), len(chunks), note)

	// 1. No-evidence short circuit.
	if (abstain || len(chunks) == 0) && deps.EmptyResponse != "" {
		logger.Printf("[Composing the answer] No supporting evidence was found; returning the configured empty response without calling the answer model.")
		return AnswerResult{Answer: deps.EmptyResponse, NoEvidence: true}
	}
	if deps.Model == nil {
		return AnswerResult{Answer: "", Failed: true, NoEvidence: len(chunks) == 0}
	}

	// 2. Build the prompt: ranked evidence under its token budget, plus the
	// question and any research findings.
	preSummary := ""
	if kb != nil {
		preSummary = kb.PreSummary
	}
	prompt := deps.answerPrompt(kb, question, partial)

	// 3. Call the model.
	callCtx, cancel := context.WithTimeout(ctx, deadlineToDuration(answerTimeoutS))
	defer cancel()

	logger.Printf("[Formalize][pre_summary] question=%q pre_summary_len=%d evidence_len=%d",
		trunc(question, 160), len(preSummary), len(prompt.user))

	reply, err := deps.Model.Complete(callCtx, []schema.Message{
		*schema.SystemMessage(prompt.system),
		*schema.UserMessage(prompt.user),
	}, nil)
	if err != nil {
		logger.Printf("[Composing the answer] composition failed: %v", err)
		return AnswerResult{Answer: answerErrorFallback, Failed: true}
	}
	return AnswerResult{Answer: cleanAnswer(reply.Content), Partial: partial}
}

// answerPrompt is the terminal node's input: the ranked evidence under its token
// budget plus the question and any research findings. Shared by the one-shot and
// the streaming compose so both render exactly the same prompt.
type answerPrompt struct {
	system  string
	user    string
	partial bool
}

func (d AnswerDeps) answerPrompt(kb *harness.Kbinfos, question string, partial bool) answerPrompt {
	chunks := []map[string]any{}
	if kb != nil {
		chunks = kb.Chunks
	}
	ranked := rankBySimilarity(chunks)
	citeChunks := ranked
	if len(citeChunks) > citeChunkCap {
		citeChunks = citeChunks[:citeChunkCap]
	}
	if len(citeChunks) == 0 {
		citeChunks = chunks
	}
	maxTokens := d.MaxTokens
	if maxTokens <= 0 {
		maxTokens = evidenceBudgetTokens
	}
	evidence := strings.Join(prompts.KBPrompt(citeChunks, maxTokens), "\n")

	parts := []string{fmt.Sprintf("Question:\n%s\n", question)}
	preSummary := ""
	if kb != nil {
		preSummary = kb.PreSummary
	}
	if strings.TrimSpace(preSummary) != "" {
		parts = append(parts, fmt.Sprintf("Research findings (authoritative — use these facts verbatim):\n%s\n", strings.TrimSpace(preSummary)))
	}
	parts = append(parts, fmt.Sprintf("Evidence:\n%s", evidence))
	user := strings.Join(parts, "\n")
	if partial {
		user = harness.PartialAnswerPreamble + "\n\n" + user
	}
	return answerPrompt{system: d.composeSystem(), user: user, partial: partial}
}

// ComposeAnswerStream is ComposeAnswer for models that can emit incrementally:
// it renders the same prompt, forwards each piece as it arrives, and returns the
// assembled answer. A streaming failure is returned so the caller can fall back
// to the one-shot call.
func ComposeAnswerStream(ctx context.Context, deps AnswerDeps, model harness.StreamingSessionModel, kb *harness.Kbinfos, question string, partial bool, onDelta func(delta string, isThink bool) error) (AnswerResult, error) {
	logger := deps.Logger
	if logger == nil {
		logger = _LOG
	}
	if model == nil {
		return AnswerResult{Answer: "", Failed: true}, nil
	}
	chunks := []map[string]any{}
	if kb != nil {
		chunks = kb.Chunks
	}
	prompt := deps.answerPrompt(kb, question, partial)
	if len(chunks) == 0 && deps.EmptyResponse != "" {
		return AnswerResult{Answer: deps.EmptyResponse, NoEvidence: true}, nil
	}

	callCtx, cancel := context.WithTimeout(ctx, deadlineToDuration(answerTimeoutS))
	defer cancel()

	reply, err := model.StreamComplete(callCtx, []schema.Message{
		*schema.SystemMessage(prompt.system),
		*schema.UserMessage(prompt.user),
	}, nil, func(delta string, isThink bool) error {
		if onDelta == nil || delta == "" {
			return nil
		}
		return onDelta(delta, isThink)
	})
	if err != nil {
		logger.Printf("[Composing the answer] streaming composition failed: %v", err)
		return AnswerResult{Answer: "", Failed: true}, err
	}
	if reply == nil {
		return AnswerResult{Answer: "", Failed: true}, errors.New("streaming composition returned no reply")
	}
	return AnswerResult{Answer: cleanAnswer(reply.Content), Partial: partial}, nil
}

// composeSystem builds the system prompt, mirroring Python's precedence rules.
//
// LANGUAGE IS DELIBERATELY LEFT OVERRIDABLE: "answer in the same language as the
// question" is exactly the rule a user setting "answer in English" means to
// replace, so it must not be listed as protected. What stays protected is the
// evidence contract: citing sources, answering the exact attribute asked for,
// and never substituting prior knowledge for missing evidence.
func (d AnswerDeps) composeSystem() string {
	// Mirror Python compose_system: the citation rules are citation_prompt
	// (citation_prompt.md) with an optional user-defined override
	// (RAGConfig.CiteRules / user_defined_prompts).
	rules := prompts.CitationPrompt(d.CiteRules)
	system := strings.ReplaceAll(harness.FinalAnswerSystem, "{cite_rules}", rules)
	if sp := strings.TrimSpace(d.SystemPrompt); sp != "" {
		system = fmt.Sprintf("%s\n\n# Assistant configuration (set by the user)\n%s\n\nFollow the configuration above for language, tone, style, format and any other presentational instruction, including where it overrides the language rule above. Where it conflicts with the citation rules, attribute fidelity, or the requirement to answer only from the provided evidence, those three take precedence.",
			system, sp)
	}
	return system
}

// cleanAnswer strips a leading thinking preamble from the composed answer.
func cleanAnswer(s string) string {
	return strings.TrimSpace(reAnswerThink.ReplaceAllString(s, ""))
}

// rankBySimilarity mirrors Python's sorted(..., key=similarity or score,
// reverse=True). Stable, so equal-scored chunks keep their retrieval order.
func rankBySimilarity(chunks []map[string]any) []map[string]any {
	out := append([]map[string]any(nil), chunks...)
	sort.SliceStable(out, func(i, j int) bool {
		return similarityOf(out[i]) > similarityOf(out[j])
	})
	return out
}

func similarityOf(c map[string]any) float64 {
	if v, ok := toFloat(c["similarity"]); ok {
		return v
	}
	if v, ok := toFloat(c["score"]); ok {
		return v
	}
	return 0.0
}

// naiveAnswerSystem is the fixed system prompt Python _naive_rag sends
// (line 1560). It is deliberately NOT FinalAnswerSystem: the naive path is a
// plain single-pass answer and carries neither the citation contract nor the
// partial-answer preamble the agentic modes compose.
const naiveAnswerSystem = "Answer the question using ONLY the numbered evidence below. Cite with [n] markers. If the evidence does not answer it, say so plainly — do not use outside knowledge."

// naiveAnswerTemperature is the sampling temperature of the naive answer call
// (Python 1556: answer_conf = gen_conf or {"temperature": 0.3}). Go's RunRequest
// carries no per-call generation config, so this is always the no-gen_conf
// branch.
const naiveAnswerTemperature = 0.3

// naiveFallbackChars caps the evidence echoed back when the naive answer call
// fails (Python 1568: evidence[:4000]).
const naiveFallbackChars = 4000

// ComposeNaiveAnswer mirrors Python _naive_rag's answer composition
// (lines 1555-1568): one retrieve pass, then a single composed answer over the
// top chunks.
//
// Unlike ComposeAnswer this does NOT use kb_prompt and does NOT use
// FinalAnswerSystem — Python renders a flat "[i] content" list truncated to the
// first 1500 chars of each chunk, capped at 8 chunks, and sends it under a
// short fixed system prompt.
//
// The messages are run through message_fit_in (Python 1561) against
// deps.MaxLength before the call.
func ComposeNaiveAnswer(ctx context.Context, deps AnswerDeps, chunks []map[string]any, question string) AnswerResult {
	logger := deps.Logger
	if logger == nil {
		logger = _LOG
	}
	if len(chunks) == 0 {
		if deps.EmptyResponse != "" {
			return AnswerResult{Answer: deps.EmptyResponse, NoEvidence: true}
		}
		return AnswerResult{NoEvidence: true}
	}

	// Python 1555: a flat "[i] content" list over the first 8 chunks, each
	// truncated to 1500 characters.
	var parts []string
	for i, c := range chunks {
		if i >= naiveEvidenceChunkCap {
			break
		}
		content := harness.ChunkTextOf(c)
		if len(content) > naiveEvidenceCharCap {
			content = content[:naiveEvidenceCharCap]
		}
		parts = append(parts, fmt.Sprintf("[%d] %s", i+1, content))
	}
	evidence := strings.Join(parts, "\n\n")
	fallback := evidence
	if len(fallback) > naiveFallbackChars {
		fallback = fallback[:naiveFallbackChars]
	}

	if deps.Model == nil {
		// Python has no model guard here, but calling a nil model would panic;
		// return the evidence the same way a failed call does.
		return AnswerResult{Answer: fallback, Failed: true}
	}

	callCtx, cancel := context.WithTimeout(ctx, deadlineToDuration(answerTimeoutS))
	defer cancel()

	// Python 1561: message_fit_in(form_message(system, user), max_length).
	fitted, fitErr := chat.FitMessages(naiveAnswerSystem, []schema.Message{
		*schema.UserMessage(fmt.Sprintf("Question: %s\n\nEvidence:\n%s", question, evidence)),
	}, deps.MaxLength)
	if fitErr != "" {
		logger.Printf("[Naive RAG] prompt fitting failed: %s", fitErr)
		return AnswerResult{Answer: fallback, Failed: true}
	}

	// Python 1562: async_chat(msg[0]["content"], msg[1:], answer_conf) — the
	// fitted system text is the first entry and the rest is the history.
	// Splitting explicitly keeps the call shaped like Python's, even though
	// Go's Complete takes the two parts re-joined.
	system := naiveAnswerSystem
	history := fitted
	if len(fitted) > 0 && fitted[0].Role == schema.System {
		system = fitted[0].Content
		history = fitted[1:]
	}
	messages := make([]schema.Message, 0, 1+len(history))
	messages = append(messages, *schema.SystemMessage(system))
	messages = append(messages, history...)

	reply, err := modelWithTemperature(deps.Model, naiveAnswerTemperature).Complete(callCtx, messages, nil)
	if err != nil {
		// Python 1567-1568: on failure yield the raw evidence, not an error.
		logger.Printf("[Naive RAG] composition failed: %v", err)
		return AnswerResult{Answer: fallback, Failed: true}
	}
	// Python 1565: str(ans or "").strip() or empty_response — an empty answer
	// degrades to the configured empty response.
	answer := cleanAnswer(reply.Content)
	if answer == "" {
		answer = deps.EmptyResponse
	}
	return AnswerResult{Answer: answer}
}

// modelWithTemperature returns a view of mdl that samples at temp, falling back
// to the model's default when it does not support per-call temperature.
func modelWithTemperature(mdl harness.SessionModel, temp float64) harness.SessionModel {
	if tm, ok := mdl.(harness.TemperatureModel); ok {
		return &temperatureModel{mdl: tm, temp: temp}
	}
	return mdl
}

type temperatureModel struct {
	mdl  harness.TemperatureModel
	temp float64
}

func (m *temperatureModel) Complete(ctx context.Context, messages []schema.Message, tools []harness.ToolSpec) (*harness.ModelReply, error) {
	return m.mdl.CompleteWithTemperature(ctx, messages, tools, m.temp)
}
