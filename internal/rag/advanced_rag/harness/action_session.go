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
	"crypto/rand"
	"encoding/json"
	"fmt"
	"math"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/cloudwego/eino/schema"
	"gorm.io/gorm"

	"ragflow/internal/agent/chat"
	"ragflow/internal/rag/prompts"

	"ragflow/internal/common"
)

// action_session.go consolidates the Go port of Python harness/action_session.py
// that was previously split across sessiontypes.go, extract.go, toolschema.go,
// run.go, navchain.go, and session.go. The split is preserved as clearly marked
// section banners below; every symbol keeps its original name and behavior so
// tool_executor.go and the tests are unaffected.
//
// The section ORDER mirrors Python action_session.py's source order
// (helpers → classes → tool specs → executors → model seam → _SessionState →
// nodes → nav prefix → entry points at the end), not a Go dependency order.

// ===================== sessiontypes.go =====================
// Slot-table models and the unified tool-result
//
// Mirrors Python harness/action_session.py:
//   - Variable / State / Result  (lines 80-149)
//   - tool status constants      (lines 157-162)
//   - ToolOutcome                (lines 165-183)
//   - apply_patch                (line 574)

// Variable is one unknown entity to resolve. ID is immutable across patches.
type Variable struct {
	ID                int
	Type              string
	QuestionClues     []string
	DiscoveredClues   []string
	Candidate         *string
	CandidateStrength *float64
}

// Brief mirrors Python Variable.brief: one-line rendering for prompts.
func (v Variable) Brief() string {
	if v.Candidate != nil && *v.Candidate != "" {
		cs := "?"
		if v.CandidateStrength != nil {
			cs = fmt.Sprintf("%.2f", *v.CandidateStrength)
		}
		return fmt.Sprintf("[%d] %s: %s (%s)", v.ID, v.Type, *v.Candidate, cs)
	}
	return fmt.Sprintf("[%d] %s: EMPTY", v.ID, v.Type)
}

// Filled mirrors Python Variable.filled.
func (v Variable) Filled() bool { return v.Candidate != nil && *v.Candidate != "" }

// State is the slot table carried through one action session.
type State struct {
	State                []Variable
	Depth                int
	ID                   string
	RetrievedEvidenceIDs []string
}

// NewState builds a State, generating its ID the way Python's
// __post_init__ does: "<depth:03x>_<millis%1e8:08x><1 random byte:02x>".
func NewState(vars []Variable, depth int, evidenceIDs []string) State {
	s := State{
		State:                vars,
		Depth:                depth,
		RetrievedEvidenceIDs: evidenceIDs,
	}
	if s.ID == "" {
		s.ID = newStateID(depth)
	}
	return s
}

// newStateID mirrors Python State.__post_init__'s id scheme.
func newStateID(depth int) string {
	ms := time.Now().UnixMilli() % 100000000
	var b [1]byte
	_, _ = rand.Read(b[:])
	return fmt.Sprintf("%03x_%08x%02x", depth, ms, b[0])
}

// Unresolved mirrors Python State.unresolved: slots still missing a candidate.
func (s State) Unresolved() []Variable {
	out := make([]Variable, 0, len(s.State))
	for _, v := range s.State {
		if !v.Filled() {
			out = append(out, v)
		}
	}
	return out
}

// ByID mirrors Python State.by_id. Returns nil when no slot carries vid.
func (s *State) ByID(vid int) *Variable {
	for i := range s.State {
		if s.State[i].ID == vid {
			return &s.State[i]
		}
	}
	return nil
}

// Brief mirrors Python State.brief: e.g. "d0(++. )".
func (s State) Brief() string {
	var b strings.Builder
	b.WriteString(fmt.Sprintf("d%d(", s.Depth))
	for _, v := range s.State {
		if v.Filled() {
			b.WriteByte('+')
		} else {
			b.WriteByte('.')
		}
	}
	b.WriteByte(')')
	return b.String()
}

// RenderSlots mirrors Python State.render_slots: the multi-line prompt block
// listing every slot, its clues, and any candidate found so far.
func (s State) RenderSlots() string {
	var lines []string
	for _, v := range s.State {
		line := fmt.Sprintf("- id=%d type=%s", v.ID, v.Type)
		if len(v.QuestionClues) > 0 {
			line += "\n  question_clues: " + strings.Join(v.QuestionClues, "; ")
		}
		if len(v.DiscoveredClues) > 0 {
			tail := v.DiscoveredClues
			if len(tail) > 4 {
				tail = tail[len(tail)-4:]
			}
			line += "\n  discovered_clues: " + strings.Join(tail, "; ")
		}
		if v.Candidate != nil && *v.Candidate != "" {
			cs := "?"
			if v.CandidateStrength != nil {
				cs = fmt.Sprintf("%.2f", *v.CandidateStrength)
			}
			line += fmt.Sprintf("\n  CANDIDATE: %s (strength=%s)", *v.Candidate, cs)
		}
		lines = append(lines, line)
	}
	return strings.Join(lines, "\n")
}

// Result is the outcome of ONE run_action session (one of NewStates /
// FoundAnswer).
//
// Messages is typed as []schema.Message rather than Python's untyped list: the
// session builds them through the eino constructors, and a typed slice keeps
// the conversation inspectable (and testable) without type assertions.
type Result struct {
	Messages             []schema.Message
	NewStates            []State
	FoundAnswer          *string
	RetrievedEvidenceIDs []string
	TerminalType         *string
	TerminalPayload      map[string]any
}

// ApplyPatch mirrors Python apply_patch (action_session.py:574): ONLY existing
// ids are patchable; the mutable fields are candidate / candidate_strength /
// discovered_clues. ID is immutable, so patches may not add variables.
//
// Returns nil when a patch entry is malformed (not a map, or missing "id") or
// when nothing actually changed — both mirror Python's early returns.
func ApplyPatch(base State, branchPatches []map[string]any) *State {
	newVars := make([]Variable, 0, len(base.State))
	for _, v := range base.State {
		nv := Variable{
			ID:                v.ID,
			Type:              v.Type,
			QuestionClues:     append([]string(nil), v.QuestionClues...),
			DiscoveredClues:   append([]string(nil), v.DiscoveredClues...),
			Candidate:         v.Candidate,
			CandidateStrength: v.CandidateStrength,
		}
		newVars = append(newVars, nv)
	}

	changed := false
	for _, pv := range branchPatches {
		if pv == nil {
			return nil
		}
		rawID, ok := pv["id"]
		if !ok {
			return nil
		}
		// Python compares `nv.id == pv["id"]` with ints, so a JSON string id
		// ("1") simply matches no slot and is skipped. Only JSON numbers match,
		// which decode as float64 — accept those, and reject anything else so the
		// behaviour is identical rather than silently more permissive.
		want, ok := toIntStrict(rawID)
		if !ok {
			continue
		}
		idx := -1
		for i := range newVars {
			if newVars[i].ID == want {
				idx = i
				break
			}
		}
		if idx < 0 {
			continue
		}
		nv := &newVars[idx]

		if raw, has := pv["candidate"]; has {
			// Mirror Python apply_patch: `str(x) if x else None` — any falsy
			// value (0, 0.0, False, "", [], {}, None) becomes None, NOT its
			// string form. fmt.Sprint would otherwise turn 0/False into
			// "0"/"false", which Python drops.
			if !isTruthy(raw) {
				nv.Candidate = nil
			} else {
				s := fmt.Sprint(raw)
				nv.Candidate = &s
			}
			changed = true
		}
		if raw, has := pv["candidate_strength"]; has && raw != nil {
			if f, ok := toFloat(raw); ok {
				v := math.Min(math.Max(f, 0.0), 1.0)
				nv.CandidateStrength = &v
				changed = true
			}
			// Unparseable strengths are ignored, mirroring Python's except/pass.
		}
		if raw, has := pv["discovered_clues"]; has {
			if list, ok := raw.([]any); ok {
				tail := list
				if len(tail) > 4 {
					tail = tail[len(tail)-4:]
				}
				for _, c := range tail {
					nv.DiscoveredClues = append(nv.DiscoveredClues, truncateRunes(fmt.Sprint(c), 160))
				}
				changed = true
			}
		}
	}
	if !changed {
		return nil
	}
	out := State{
		State:                newVars,
		Depth:                base.Depth + 1,
		RetrievedEvidenceIDs: append([]string(nil), base.RetrievedEvidenceIDs...),
	}
	out.ID = newStateID(out.Depth)
	return &out
}

// truncateRunes cuts s to at most n characters (runes, not bytes), matching
// Python's str slicing semantics.
func truncateRunes(s string, n int) string {
	if len([]rune(s)) <= n {
		return s
	}
	return string([]rune(s)[:n])
}

// toInt coerces a JSON-decoded value to int. Handles float64 (the default for
// JSON numbers), the integer types, and numeric strings.
func toInt(v any) (int, bool) {
	switch n := v.(type) {
	case int:
		return n, true
	case int32:
		return int(n), true
	case int64:
		return int(n), true
	case float32:
		return int(n), true
	case float64:
		return int(n), true
	case string:
		if i, err := strconv.Atoi(strings.TrimSpace(n)); err == nil {
			return i, true
		}
	}
	return 0, false
}

// toIntStrict converts JSON NUMBER types only, mirroring Python's strict
// integer comparison. Unlike toInt it does NOT parse strings: a patch whose id
// is "1" must match nothing, exactly as in Python.
func toIntStrict(v any) (int, bool) {
	switch n := v.(type) {
	case int:
		return n, true
	case int32:
		return int(n), true
	case int64:
		return int(n), true
	case float32:
		return int(n), true
	case float64:
		return int(n), true
	}
	return 0, false
}

// toFloat coerces a JSON-decoded value to float64.
func toFloat(v any) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case float32:
		return float64(n), true
	case int:
		return float64(n), true
	case int32:
		return float64(n), true
	case int64:
		return float64(n), true
	case string:
		return ParseFloat(n)
	}
	return 0, false
}

// ParseFloat parses a float64 from a string, tolerating surrounding
// whitespace. Exported for the orchestrator package, whose SCA verdict
// coercion needs the same lenient parsing.
func ParseFloat(s string) (float64, bool) {
	f, err := strconv.ParseFloat(strings.TrimSpace(s), 64)
	if err != nil {
		return 0, false
	}
	return f, true
}

// isTruthy mirrors Python's bool() truthiness for the JSON-decoded values that
// reach apply_patch: None/nil, empty string/collection, zero number, and False
// are falsy; everything else (including non-empty objects and any other type) is
// truthy. Used so `str(x) if x else None` behaves identically in Go.
func isTruthy(v any) bool {
	switch x := v.(type) {
	case nil:
		return false
	case bool:
		return x
	case string:
		return x != ""
	case int:
		return x != 0
	case int32:
		return x != 0
	case int64:
		return x != 0
	case float32:
		return x != 0
	case float64:
		return x != 0
	case []any:
		return len(x) > 0
	case []string:
		return len(x) > 0
	case map[string]any:
		return len(x) > 0
	default:
		return true
	}
}

// ===================== extract.go =====================
// JSON / terminal-block extraction from model output.
//
// Mirrors Python action_session.py::extract_json (line 512) and extract_tag
// (line 551). Models wrap JSON in thinking preamble and Markdown fences, and
// often emit a second object after the first — a greedy regex over-captures
// there and the parse then fails with "Extra data", so the extractor
// brace-matches to isolate ONE complete object and walks to the next "{" when
// that one is invalid.

// JSON / terminal-block extraction from model output (mirrors Python
// action_session.py::extract_json / extract_tag).

// ExtractJSON returns the first parseable JSON value found in text, or nil.
func ExtractJSON(text string) any {
	if text == "" {
		return nil
	}
	n := len(text)
	for i := 0; i < n; {
		start := strings.IndexByte(text[i:], '{')
		if start < 0 {
			return nil
		}
		start += i
		depth := 0
		for j := start; j < n; j++ {
			switch text[j] {
			case '{':
				depth++
			case '}':
				depth--
				if depth == 0 {
					candidate := text[start : j+1]
					var out any
					if err := json.Unmarshal([]byte(candidate), &out); err == nil {
						return out
					}
					// Python extract_json first tries json.loads(strict=False),
					// then json_repair.loads(candidate); only if BOTH fail does it
					// break to the next "{". Mirror that leniency here.
					if v := repairJSONObject(candidate); v != nil {
						return v
					}
					// Invalid object; try the next "{".
				}
			}
		}
		i = start + 1
	}
	return nil
}

// repairJSONObject is a conservative stand-in for Python's json_repair.loads,
// applied only after a candidate fails strict json.Unmarshal. It walks cheap,
// idempotent normalizations that models commonly emit and re-validates after
// each: trailing commas, single-quoted strings, NaN/Infinity numeric literals,
// and illegal control characters inside string literals (which Python's
// json.loads(strict=False) and json_repair both tolerate). It never rewrites
// already-valid JSON and returns nil when no transform yields a parse.
func repairJSONObject(candidate string) any {
	steps := []struct {
		name  string
		apply func(string) string
	}{
		{"dropTrailingCommas", dropTrailingCommas},
		{"normalizeSingleQuotes", normalizeSingleQuotes},
		{"nullNonFinite", nullNonFiniteLiterals},
		{"stripRawControlChars", stripRawControlChars},
	}
	fixed := candidate
	for _, s := range steps {
		fixed = s.apply(fixed) // accumulate so multi-step repairs compose
		var out any
		if err := json.Unmarshal([]byte(fixed), &out); err == nil {
			return out
		}
	}
	return nil
}

// dropTrailingCommas removes a comma immediately before a closing brace or
// bracket (e.g. {"a": 1,} or [1, 2,]), repeated until stable.
func dropTrailingCommas(s string) string {
	for {
		before := s
		s = strings.ReplaceAll(s, ",}", "}")
		s = strings.ReplaceAll(s, ",]", "]")
		if s == before {
			return s
		}
	}
}

// normalizeSingleQuotes rewrites single-quoted string literals to double-quoted
// ones, mirroring json_repair. It only touches quotes outside of double-quoted
// strings so it never corrupts existing valid JSON.
func normalizeSingleQuotes(s string) string {
	var b strings.Builder
	inDouble := false
	inSingle := false
	escaped := false
	for i := 0; i < len(s); i++ {
		ch := s[i]
		if escaped {
			b.WriteByte(ch)
			escaped = false
			continue
		}
		switch ch {
		case '\\':
			escaped = true
			b.WriteByte(ch)
		case '"':
			if !inSingle {
				inDouble = !inDouble
			}
			b.WriteByte(ch)
		case '\'':
			if !inDouble {
				inSingle = !inSingle
				b.WriteByte('"') // replace single quote with double quote
			} else {
				b.WriteByte(ch)
			}
		default:
			b.WriteByte(ch)
		}
	}
	return b.String()
}

// nullNonFiniteLiterals rewrites bare NaN / Infinity / -Infinity tokens (which
// Python's json_repair normalizes) into JSON null so the object can parse.
func nullNonFiniteLiterals(s string) string {
	s = strings.ReplaceAll(s, ": NaN", ": null")
	s = strings.ReplaceAll(s, ": Infinity", ": null")
	s = strings.ReplaceAll(s, ": -Infinity", ": null")
	return s
}

// stripRawControlChars removes raw ASCII control characters (other than tab,
// newline and carriage-return) that would otherwise make Go's json.Unmarshal
// reject a string literal; Python's json.loads(strict=False) accepts them.
func stripRawControlChars(s string) string {
	return strings.Map(func(r rune) rune {
		if r < 0x20 && r != '\t' && r != '\n' && r != '\r' {
			return -1
		}
		return r
	}, s)
}

var reFencedJSON = regexp.MustCompile("(?s)```(?:json)?\\s*(\\{.*?\\}|\\[.*?\\])\\s*```")

// ExtractTag mirrors Python extract_tag: exact-tag extraction first, then
// lenient fallbacks for models that wrap the JSON in code fences or emit bare
// objects (observed with DeepSeek-class models ignoring the XML protocol).
//
// Returns "" when the tag is absent.
func ExtractTag(text, tag string) string {
	if text == "" {
		return ""
	}
	open, closeTag := "<"+tag+">", "</"+tag+">"
	s := strings.LastIndex(text, open)
	e := strings.LastIndex(text, closeTag)
	if s >= 0 && e > s {
		return strings.TrimSpace(text[s+len(open) : e])
	}
	// Fenced ```json ... ``` blocks mentioning the tag name.
	fenced := reFencedJSON.FindAllStringSubmatch(text, -1)
	if len(fenced) == 0 {
		if m, _ := regexp.MatchString(`<`+regexp.QuoteMeta(tag)+`\b`, text); !m {
			return ""
		}
		return ""
	}
	want := map[string]bool{"new_states": true}
	if tag != "state" {
		want = map[string]bool{"answer": true, "new_state": true}
	}
	// Prefer the LAST block (most recent decision).
	for i := len(fenced) - 1; i >= 0; i-- {
		var obj map[string]any
		if err := json.Unmarshal([]byte(fenced[i][1]), &obj); err != nil {
			continue
		}
		for k := range obj {
			if want[k] {
				return strings.TrimSpace(fenced[i][1])
			}
		}
	}
	return ""
}

// ExtractJSONObject parses a JSON object from text, returning nil when the text
// is empty, unparseable, or does not decode to an object. Used for tool payloads
// that arrive as raw JSON strings.
func extractJSONObject(text string) any {
	if strings.TrimSpace(text) == "" {
		return nil
	}
	if v := ExtractJSON(text); v != nil {
		if _, ok := v.(map[string]any); ok {
			return v
		}
	}
	var out map[string]any
	if err := json.Unmarshal([]byte(strings.TrimSpace(text)), &out); err != nil {
		return nil
	}
	return out
}

// ===================== toolschema.go =====================
// Tool schemas, the per-session tool object, and the two seams the action
// session depends on.
//
// Mirrors Python harness/action_session.py:
//   - tool specs        (lines 187-437)
//   - _TOOL_MAP         (line 428)
//   - _active_tool_specs(line 440)
//   - _disable_tool     (line 472)
//   - _reason_status    (line 495)
//   - execute_tool      (line 1003)

func arrayParam(desc string, minItems, maxItems int) map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"query": map[string]any{
				"type":     "array",
				"items":    map[string]any{"type": "string"},
				"minItems": minItems,
				"maxItems": maxItems,
			},
		},
		"required": []string{"query"},
	}
}

// Tool schemas. Descriptions are copied verbatim from Python: they are the
// model's only guide for when to pick which tool, and rephrasing them changes
// routing behaviour.
var (
	retrieveToolSpec = ToolSpec{
		Type: "function",
		Function: ToolFunction{
			Name: "retrieve",
			Description: ("Keyword-first search of the fixed document corpus. Pass natural-" +
				"language queries; returns SHORT snippets of the most relevant " +
				"passages (exact-term matched where possible). Use multiple queries " +
				"to cover different aspects. Supports 1-3 queries per call."),
			Parameters: arrayParam("", 1, 3),
		},
	}

	listChunksToolSpec = ToolSpec{
		Type: "function",
		Function: ToolFunction{
			Name: "list_chunks",
			Description: ("Deep-read the FULL text of one document by doc_id (returned in " +
				"retrieve snippets). Use for enumeration / count / arithmetic answers " +
				"when snippets are insufficient. Returns all chunks of the document " +
				"in reading order. One doc_id per call."),
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"doc_id": map[string]any{
						"type":        "string",
						"description": "document id seen in a retrieve snippet",
					},
				},
				"required": []string{"doc_id"},
			},
		},
	}

	searchChunksToolSpec = ToolSpec{
		Type: "function",
		Function: ToolFunction{
			Name: "search_chunks",
			Description: ("SEMANTIC retrieval (hybrid vector+BM25) with COMPILED-STRUCTURE " +
				"EXPANSION. Use as the PRIMARY recall tool when exact-term " +
				"``retrieve`` returns nothing useful, or when the dataset is large and " +
				"you are unsure which document holds the answer — the answer passage " +
				"may share NO surface words with the query. " +
				"Compiled expansion: automatically appends related chunks from the " +
				"dataset's compiled structure (page index, tree/heading hierarchy, " +
				"knowledge graph, wiki pages when present) so a semantic hit carries " +
				"its structural neighbours (parent/child headings, sibling pages). " +
				"If the dataset has NO compiled structure (incl. no wiki), expansion " +
				"is a no-op — no error, just semantic hits. " +
				"Returns snippet chunks ranked by relevance. 1-2 queries per call."),
			Parameters: arrayParam("", 1, 2),
		},
	}

	webSearchToolSpec = ToolSpec{
		Type: "function",
		Function: ToolFunction{
			Name: "web_search",
			Description: ("Search the open WEB. Use ONLY when the needed fact is world " +
				"knowledge / recent event / not covered by the fixed corpus — e.g. " +
				"a current event, a person's alive-now status, or a statistic newer " +
				"than the corpus. If the fact plausibly lives in the documents, " +
				"prefer corpus tools (retrieve/search_chunks) first. 1-2 queries per call."),
			Parameters: arrayParam("", 1, 2),
		},
	}

	// wikiQueryToolSpec mirrors Python tools/exploration.py wiki_query: a hybrid
	// search over the compiled "wiki_page_draft" rows. It is an UNPLUGGED
	// extension seam — ported faithfully but never surfaced (not in allTools and
	// no SearchWiki backend is wired in production), matching Python, which keeps
	// the function but never registers it in the action session surface.
	wikiQueryToolSpec = ToolSpec{
		Type: "function",
		Function: ToolFunction{
			Name: "wiki_query",
			Description: ("Search the COMPILED WIKI PAGES when the raw corpus is insufficient. " +
				"Use it for synthesised, narrative explanations derived from the uploaded " +
				"documents — the wiki already condenses many chunks into one page, so a " +
				"question is often answerable straight from it. Prefer corpus tools " +
				"(retrieve/search_chunks) first when the fact plausibly lives in the raw text."),
			Parameters: arrayParam("One free-form question for the wiki; optional second element: comma-separated keywords to bias the page ranking.", 1, 2),
		},
	}

	navigateTreeToolSpec = ToolSpec{
		Type: "function",
		Function: ToolFunction{
			Name: "navigate_tree",
			Description: ("LOCATE the RIGHT DOCUMENT among MANY before deep-reading. Use it " +
				"BEFORE search_chunks when the dataset is large and you have no " +
				"doc_id yet — it routes by TOPIC/CLUSTERING similarity over the " +
				"compiled document-navigation tree (not exact surface words), so it " +
				"finds the document even when your query words differ from its text. " +
				"Returns candidate doc_ids + a first-chunk summary of each. " +
				"This is the FIRST hop of a navigation chain: " +
				"navigate_tree(query) -> doc_id -> navigate_structure(doc_id, ...) " +
				"-> list_chunks(doc_id, chunk_ids). " +
				"Use when: the question names a topic/entity/alias but you do not " +
				"know which document discusses it; search_chunks returned scattered " +
				"hits across many docs and you must pick the source. " +
				"Do NOT use if you already hold a doc_id (go straight to " +
				"navigate_structure) or if the answer is likely a single exact " +
				"passage (prefer retrieve/search_chunks). " +
				"If the dataset has no compiled document navigation tree, it returns " +
				"empty — fall back to search_chunks."),
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"query": map[string]any{
						"type":        "string",
						"description": "topic / entity / alias whose document(s) to locate",
					},
				},
				"required": []string{"query"},
			},
		},
	}

	navigateStructureToolSpec = ToolSpec{
		Type: "function",
		Function: ToolFunction{
			Name: "navigate_structure",
			Description: ("PINPOINT A PASSAGE inside ONE document using its compiled structure " +
				"(heading/catalog tree, concept mindmap, or entity graph) — the " +
				"in-document counterpart of navigate_tree. " +
				"Use AFTER you know the doc_id (from navigate_tree / search_chunks / " +
				"retrieve) and need to find where the answer lives WITHOUT reading " +
				"every chunk. Returns the structure outline annotated with matching " +
				"chunk_ids (reading-order aware). Then call list_chunks(doc_id, " +
				"chunk_ids) to read exactly those. " +
				"kind: 'catalog' (default) for page-index/heading/timeline trees, " +
				"'mindmap' for concept maps, 'graph' for entity-relation graphs. " +
				"If the document has NO compiled structure, an empty <doc/> is " +
				"returned — fall back to list_chunks to read the full document."),
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"doc_id": paramString("document id seen from a prior tool result"),
					"query":  paramString("what to locate within the document"),
					"kind":   paramEnum("compiled structure kind, default catalog", "catalog", "mindmap", "graph"),
				},
				"required": []string{"doc_id"},
			},
		},
	}

	calculateToolSpec = ToolSpec{
		Type: "function",
		Function: ToolFunction{
			Name: "calculate",
			Description: ("COMPUTE a numeric answer by generating and safely running code. " +
				"MANDATORY whenever the question asks you to DERIVE a number by " +
				"combining facts you found (sum/difference/percentage/ratio/sort/" +
				"compare/difference in length/age, price, area, growth, etc.) — do " +
				"NOT do arithmetic mentally. Language-neutral: the question and " +
				"facts may be in ANY language (English, Chinese, ...); pass the " +
				"numbers verbatim as written in the evidence regardless of language. " +
				"Steps: (1) collect every needed number first (retrieve / " +
				"search_chunks / navigate_* / list_chunks); (2) call calculate with " +
				"the question + ALL those numbers; (3) report the computed result " +
				"verbatim. If a needed number is missing, search for it first — do " +
				"not estimate. If the answer IS one of the stated numbers (no " +
				"combination needed), answer directly without this tool."),
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"question": paramString("the user's question, verbatim"),
					"facts": map[string]any{
						"type":        "array",
						"items":       map[string]any{"type": "string"},
						"description": "numbers/facts found in the evidence, verbatim",
					},
				},
				"required": []string{"question", "facts"},
			},
		},
	}

	graphExploreToolSpec = ToolSpec{
		Type: "function",
		Function: ToolFunction{
			Name: "graph_explore",
			Description: ("EXPLORE the compiled KNOWLEDGE GRAPH (entities + relations) for a " +
				"RELATIONAL/multi-hop answer. Different from navigate_*: instead of " +
				"locating a document or passage, it seeds entities for the query, " +
				"hops along their RELATIONS, and returns either a direct answer or " +
				"the source passages behind the relevant entities/relations. " +
				"Use when the answer requires connecting several entities through " +
				"their relations (e.g. who-was-related-to-whom, cause-effect chains, " +
				"membership/ownership) and you already have a starting entity from a " +
				"search result, a navigation outline, or a list_chunks reading. " +
				"If the dataset has NO compiled knowledge graph, it returns empty — " +
				"fall back to search_chunks / navigate_structure."),
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"query": paramString("the relational question / starting entity"),
					"doc_scope": map[string]any{
						"type":        "array",
						"items":       map[string]any{"type": "string"},
						"description": "optional doc_ids to restrict the graph to (from prior navigation/list_chunks)",
					},
				},
				"required": []string{"query"},
			},
		},
	}
)

func paramString(desc string) map[string]any {
	return map[string]any{"type": "string", "description": desc}
}

func paramEnum(desc string, values ...string) map[string]any {
	return map[string]any{"type": "string", "enum": values, "description": desc}
}

// ToolMap is the multi-tool registry, mirroring Python _TOOL_MAP.
// executeTool dispatches by name; add a tool by registering its schema here.
var ToolMap = map[string]ToolSpec{
	"retrieve":           retrieveToolSpec,
	"search_chunks":      searchChunksToolSpec,
	"list_chunks":        listChunksToolSpec,
	"navigate_tree":      navigateTreeToolSpec,
	"navigate_structure": navigateStructureToolSpec,
	"calculate":          calculateToolSpec,
	"graph_explore":      graphExploreToolSpec,
	"web_search":         webSearchToolSpec,
	"wiki_query":         wikiQueryToolSpec,
}

// ToolMapNames are the registered tool names, sorted for deterministic output.
func ToolMapNames() []string {
	names := make([]string, 0, len(ToolMap))
	for n := range ToolMap {
		names = append(names, n)
	}
	sort.Strings(names)
	return names
}

// ---------------------------------------------------------------------------
// The per-session tool object
// ---------------------------------------------------------------------------

// ToolExecutor runs one tool call by name and reports what happened.
//
// Every executor returns a ToolOutcome so the tool node can act on the KIND of
// result (empty / miss / poor / redundant / error) instead of measuring payload
// size.
type ToolExecutor interface {
	Execute(ctx context.Context, name string, args map[string]any) (ToolOutcome, error)
}

// Toolset is the per-session tool object, mirroring the Python RAGTools
// surface the session reads: thinking mode, web provider availability, the
// runtime-disabled tool set, and the executor.
//
// Python mutates `tools._disabled_tools` on the request object; Go keeps that
// state here so a session cannot leak it into another request.
type Toolset struct {
	// ThinkingMode selects the ModeSpec (see config.go).
	ThinkingMode string
	// HasWebSearch reports whether a web provider is configured. When false,
	// web_search is hidden from the surface rather than merely discouraged.
	HasWebSearch bool
	// DisabledTools holds tools proven unavailable this session (no compiled
	// structure of their kind).
	DisabledTools map[string]bool
	// Exec runs the tools.
	Exec ToolExecutor
}

// GetThinkingMode implements ThinkingModeCarrier so ResolveMode works on *Toolset.
func (t *Toolset) GetThinkingMode() string {
	if t == nil {
		return ""
	}
	return t.ThinkingMode
}

// ActiveToolSpecs mirrors Python _active_tool_specs: the tool schemas exposed
// to the model for THIS mode.
//
// Visibility rules:
//   - the per-mode tool set comes from config.THINKING_MODES (only ultra sees
//     the relational graph_explore);
//   - web_search is hidden when no web provider is configured — the model
//     otherwise retries it several times per session burning turns;
//   - compile-only tools stay visible even on datasets without compiled
//     structure: their executors return an explicit empty-with-hint so the
//     model falls back to plain search;
//   - RUNTIME: a compile-only tool is removed after it proves the dataset has
//     no such compiled structure (DisableTool).
func (t *Toolset) ActiveToolSpecs() []ToolSpec {
	spec := ResolveMode(t)
	var out []ToolSpec
	for _, name := range spec.ToolNames() {
		if name == "web_search" && !t.HasWebSearch {
			continue
		}
		if t.DisabledTools[name] {
			continue
		}
		if s, ok := ToolMap[name]; ok {
			out = append(out, s)
		}
	}
	return out
}

// DisableTool mirrors Python _disable_tool: mark a compile-only tool
// unavailable for the REST of this session, so ActiveToolSpecs stops
// advertising it and ExecuteTool short-circuits it.
func (t *Toolset) DisableTool(name string) {
	if _, ok := ToolMap[name]; !ok {
		return
	}
	if t.DisabledTools == nil {
		t.DisabledTools = map[string]bool{}
	}
	t.DisabledTools[name] = true
}

// IsDisabled reports whether name has been disabled this session.
func (t *Toolset) IsDisabled(name string) bool {
	return t.DisabledTools[name]
}

// ReasonStatus mirrors Python _reason_status: single source of truth for the
// cause→status mapping, so a tool cannot disagree with itself about what its
// own reason means.
// The unified tool-result contract shared by the session and the tool
// implementations (mirrors Python action_session.py's ToolOutcome and the
// OK/EMPTY/MISS/POOR/REDUNDANT/ERROR statuses — which Python also defines in
// action_session.py, so they live here in its Go mirror).

// Tool-result statuses. Mirrors Python OK/EMPTY/MISS/POOR/REDUNDANT/ERROR.
// These are the signal the session's tool node acts on.
const (
	StatusOK        = "ok"        // normal hit
	StatusEmpty     = "empty"     // dataset-level: no such compiled structure exists here
	StatusMiss      = "miss"      // query-level: nothing matched THIS query; tool still valid
	StatusPoor      = "poor"      // produced output, but too weak to be useful
	StatusRedundant = "redundant" // ran fine, but added no NEW evidence
	StatusError     = "error"     // infra / provider failure
)

// Machine-readable ToolOutcome reasons. Only ReasonNoStructure is
// DATASET-level and may disable a tool; ReasonNoDoc is QUERY-level and must not.
const (
	ReasonNone        = ""
	ReasonNoStructure = "no_structure"
	ReasonNoDoc       = "no_doc"
	ReasonInfra       = "infra"
	ReasonBadArgs     = "bad_args"
)

// ToolOutcome is the result of ONE tool call. Every executor returns one so the
// tool node can inspect WHAT happened instead of counting characters in a JSON
// string — timeout / budget / routing / disable policies hook into these fields.
type ToolOutcome struct {
	Payload     []any          // what the model sees (passage dicts)
	EvidenceIDs []string       // doc/chunk ids worth tracking
	Status      string         // one of the Status* constants
	Reason      string         // one of the Reason* constants
	Metrics     map[string]any // numeric signals: hits, new_evidence, top_score, ...
}

// NewToolOutcome builds an OK outcome with an empty metric map.
func NewToolOutcome(payload []any, evidenceIDs []string) ToolOutcome {
	return ToolOutcome{
		Payload:     payload,
		EvidenceIDs: evidenceIDs,
		Status:      StatusOK,
		Reason:      ReasonNone,
		Metrics:     map[string]any{},
	}
}

// ReasonStatus maps a ToolOutcome reason to the Status it implies, mirroring
// Python action_session._reason_status.
func ReasonStatus(reason string) string {
	switch reason {
	case ReasonNoStructure:
		return StatusEmpty
	case ReasonBadArgs, ReasonInfra:
		return StatusError
	}
	// ReasonNoDoc: this query reached nothing; the tool itself is fine.
	return StatusMiss
}

// ===================== session.go (model seam, _SessionState, nodes, routing) =====================
// The graph-edge action session: one bounded ReAct loop pursuing ONE direction.
//
// Mirrors Python harness/action_session.py.
//
// PYTHON USES LANGGRAPH; THIS FILE USES AN EXPLICIT LOOP. The graph is small
// and fixed —
//
//	START → run_action → (route) → tool|finalize|END|run_action
//	                   tool → (route_after_tool) → run_action|finalize
//	                   finalize → END
//
// — so compiling a graph buys nothing over three functions and a switch. The
// node bodies, the routing predicates, and their ordering are ported verbatim
// below; only the driver differs. See sessionLoop for the edge-by-edge mapping.
//
// ONE structural difference, isolated to one seam: Python calls the provider's
// native tool-calling API (`tools=[...]`, `tool_choice="auto"`) and parses
// `message.tool_calls`. The Go chat seam (internal/agent/chat) has no tools
// field, so tool invocation goes through the SessionModel interface — whose
// default implementation asks the model to emit the same call as a JSON block
// and parses it back. The session logic downstream is identical either way:
// swapping in a native implementation is a one-file change.

const (
	// initTimeoutS bounds the slot-table decomposition call.
	initTimeoutS = 45.0
	// actionTimeoutS is the default per-session wall-clock budget.
	actionTimeoutS = 75.0
	// snippetsPerQuery caps snippets returned per query.
	snippetsPerQuery = 4
	// maxToolResponseChars bounds ONE tool payload.
	maxToolResponseChars = 12000
	// emptyStrikes is how many dataset-level empties (reason=no_structure) a
	// compiled-structure tool must accumulate before being disabled for the rest
	// of the session. Kept above 1: a single empty can be SCOPED — graph_explore
	// over a doc_scope that has no knowledge graph says nothing about the other
	// documents, and disabling on the first empty is exactly the over-eager
	// behaviour this replaced.
	emptyStrikes = 2
	// nearDupJaccard is the token-overlap threshold at which a retrieval query
	// counts as a paraphrase of an already-run query. The model re-issues the
	// SAME intent with many paraphrases (observed: 20+ variants in ONE session),
	// each hitting the index even though the evidence is unchanged.
	nearDupJaccard = 0.8
	// skippedDupLimit: once this many near-duplicate retrievals were skipped,
	// further turns are unlikely to surface new evidence, so the session
	// converges to finalize instead of burning the remaining budget on
	// paraphrases (Q30/Q759 timeout root cause).
	skippedDupLimit = 2
	// finalizeTimeout bounds the salvage call in the finalize node.
	finalizeTimeoutS = 150.0
	// minFinalizeTimeout is the floor for the salvage call.
	minFinalizeTimeoutS = 15.0
)

// retrievalTools: near-duplicate suppression applies to these (Python
// _RETRIEVAL_TOOLS). grep_chunks / grep_search are legacy names kept so a
// model emitting them is still deduped rather than executed.
var retrievalTools = map[string]bool{
	"search_chunks": true,
	"grep_chunks":   true,
	"grep_search":   true,
}

var _LOG = common.StdLogger()

var reSearchToken = regexp.MustCompile(`[a-z0-9]{2,}`)

// searchTokens mirrors Python _search_tokens: lowercased alphanumeric tokens
// of a query, for near-duplicate detection.
func searchTokens(q string) map[string]struct{} {
	out := map[string]struct{}{}
	for _, t := range reSearchToken.FindAllString(strings.ToLower(q), -1) {
		out[t] = struct{}{}
	}
	return out
}

// IsNearDup mirrors Python _is_near_dup: true when q shares >=
// nearDupJaccard of its tokens with any query in seen.
func IsNearDup(q string, seen []string) bool {
	if strings.TrimSpace(q) == "" || len(seen) == 0 {
		return false
	}
	toks := searchTokens(q)
	if len(toks) < 2 {
		return false
	}
	for _, s := range seen {
		other := searchTokens(s)
		inter := 0
		for t := range toks {
			if _, ok := other[t]; ok {
				inter++
			}
		}
		union := len(toks)
		for t := range other {
			if _, ok := toks[t]; !ok {
				union++
			}
		}
		if union > 0 && float64(inter)/float64(union) >= nearDupJaccard {
			return true
		}
	}
	return false
}

// ---------------------------------------------------------------------------
// The model seam
// ---------------------------------------------------------------------------

// CompleteWithTemperature implements TemperatureModel.
func (m *InvokerSessionModel) CompleteWithTemperature(ctx context.Context, messages []schema.Message, tools []ToolSpec, temp float64) (*ModelReply, error) {
	if m == nil || m.Invoker == nil {
		return nil, fmt.Errorf("harness: no chat invoker configured")
	}
	msgs := messages
	if len(tools) > 0 {
		msgs = append([]schema.Message{}, messages...)
		msgs = append(msgs, *schema.UserMessage(toolCallInstruction(tools)))
	}
	req := chat.Request{Messages: msgs, Temperature: &temp}
	if chatTools := toChatTools(tools); chatTools != nil {
		req.Tools = chatTools
		req.ToolChoice = chat.ToolChoiceAuto
	}
	resp, err := m.Invoker.Invoke(ctx, m.DB, req)
	if err != nil {
		return nil, err
	}
	reply := &ModelReply{Content: resp.Content}
	if resp != nil {
		reply.Usage = resp.Usage
		reply.Tokens = totalOf(resp)
	}
	reply.ToolCalls = parseToolBlocks(resp.Content, tools)
	if len(resp.ToolCalls) > 0 {
		reply.ToolCalls = nativeToHarnessCalls(resp.ToolCalls, tools)
	}
	return reply, nil
}

// InvokerSessionModel adapts chat.Invoker to SessionModel using prompt-based
// tool calling.
//
// The tool schemas are rendered into the system prompt and the model answers
// with a fenced JSON block:
//
//	```tool
//	{"tool": "retrieve", "args": {"query": ["..."]}}
//	```
//
// Models that ignore it fall through to plain text, which the session treats as
// a no-call turn and nudges — the same path Python takes when a provider omits
// tool_calls.
type InvokerSessionModel struct {
	Invoker chat.Invoker
	DB      *gorm.DB
	// MaxLength is the resolved chat model's context window in tokens
	// (Python LLMBundle.max_length / tools.chat_mdl.max_length). Used by
	// ContextLength to size message_fit_in. <= 0 means "unknown", in which case
	// ContextLength reports chat.EffectiveContextLength's 8192 fallback.
	MaxLength int
}

// ContextLength implements ContextLengthModel. It returns the model's context
// window in tokens when known, otherwise the 8192 default that mirrors
// Python's LLM.max_length falling back when the config omits max_tokens.
func (m *InvokerSessionModel) ContextLength() int {
	if m == nil {
		return chat.EffectiveContextLength(0)
	}
	return chat.EffectiveContextLength(m.MaxLength)
}

// toolBlockRe extracts the last fenced tool block, preferring ```tool but
// accepting a bare ```json that carries a "tool" key.
var toolBlockRe = regexp.MustCompile("(?s)```(?:tool|json)?\\s*(\\{.*?\\})\\s*```")

// Complete implements SessionModel.
func (m *InvokerSessionModel) Complete(ctx context.Context, messages []schema.Message, tools []ToolSpec) (*ModelReply, error) {
	if m == nil || m.Invoker == nil {
		return nil, fmt.Errorf("harness: no chat invoker configured")
	}
	msgs := messages
	if len(tools) > 0 {
		msgs = append([]schema.Message{}, messages...)
		msgs = append(msgs, *schema.UserMessage(toolCallInstruction(tools)))
	}
	temp := 0.3
	req := chat.Request{Messages: msgs, Temperature: &temp}
	if chatTools := toChatTools(tools); chatTools != nil {
		req.Tools = chatTools
		req.ToolChoice = chat.ToolChoiceAuto
	}
	resp, err := m.Invoker.Invoke(ctx, m.DB, req)
	if err != nil {
		return nil, err
	}
	reply := &ModelReply{Content: resp.Content}
	if resp != nil {
		reply.Usage = resp.Usage
		reply.Tokens = totalOf(resp)
	}
	reply.ToolCalls = parseToolBlocks(resp.Content, tools)
	if len(resp.ToolCalls) > 0 {
		reply.ToolCalls = nativeToHarnessCalls(resp.ToolCalls, tools)
	}
	return reply, nil
}

// StreamComplete implements StreamingSessionModel by delegating to the
// underlying invoker when it can stream. A non-streaming invoker yields an
// error, which the caller treats as "fall back to the one-shot call" — the
// prompt is identical either way, so the answer is not affected.
func (m *InvokerSessionModel) StreamComplete(ctx context.Context, messages []schema.Message, tools []ToolSpec, onDelta func(delta string, isThink bool) error) (*ModelReply, error) {
	if m == nil || m.Invoker == nil {
		return nil, fmt.Errorf("harness: no chat invoker configured")
	}
	streamer, ok := m.Invoker.(chat.StreamingInvoker)
	if !ok {
		return nil, fmt.Errorf("harness: chat invoker %T does not support streaming", m.Invoker)
	}
	msgs := messages
	if len(tools) > 0 {
		msgs = append([]schema.Message{}, messages...)
		msgs = append(msgs, *schema.UserMessage(toolCallInstruction(tools)))
	}
	temp := 0.3
	req := chat.Request{Messages: msgs, Temperature: &temp}
	if chatTools := toChatTools(tools); chatTools != nil {
		req.Tools = chatTools
		req.ToolChoice = chat.ToolChoiceAuto
	}
	resp, err := streamer.Stream(ctx, m.DB, req, onDelta)
	if err != nil {
		return nil, err
	}
	if resp == nil {
		return nil, fmt.Errorf("harness: streaming invoker returned no response")
	}
	reply := &ModelReply{
		Content: resp.Content,
		Usage:   resp.Usage,
		Tokens:  totalOf(resp),
	}
	reply.ToolCalls = parseToolBlocks(resp.Content, tools)
	if len(resp.ToolCalls) > 0 {
		reply.ToolCalls = nativeToHarnessCalls(resp.ToolCalls, tools)
	}
	return reply, nil
}

// totalOf returns the total token count for a chat.Response, preferring the
// Usage split and falling back to the legacy single counter.
func totalOf(resp *chat.Response) int {
	if resp == nil {
		return 0
	}
	if resp.Usage != nil {
		return resp.Usage.TotalTokens
	}
	return resp.Tokens
}

// toolCallInstruction renders the available tools as a prompt instruction.
func toolCallInstruction(tools []ToolSpec) string {
	var b strings.Builder
	b.WriteString("You may call at most one tool per reply. To call one, output EXACTLY one fenced block and nothing else:\n\n```tool\n{\"tool\": \"<name>\", \"args\": { ... }}\n```\n\nAvailable tools:\n")
	for _, t := range tools {
		b.WriteString(fmt.Sprintf("- %s: %s\n", t.Function.Name, t.Function.Description))
		if raw, err := json.Marshal(t.Function.Parameters); err == nil {
			b.WriteString(fmt.Sprintf("  args schema: %s\n", raw))
		}
	}
	b.WriteString("\nIf no tool is needed, reply normally (the <state>/<answer> XML protocol still applies).")
	return b.String()
}

// toChatTools converts the harness ToolSpec surface into the chat seam's native
// Tool declarations. The chat seam forwards them to the provider as native
// tools; if the provider has no native tool support (or returns a fenced block
// instead), parseToolBlocks still recovers the call from the content.
func toChatTools(tools []ToolSpec) []chat.Tool {
	if len(tools) == 0 {
		return nil
	}
	out := make([]chat.Tool, 0, len(tools))
	for _, t := range tools {
		if t.Function.Name == "" {
			continue
		}
		out = append(out, chat.Tool{
			Type: "function",
			Function: chat.ToolFunction{
				Name:        t.Function.Name,
				Description: t.Function.Description,
				Parameters:  t.Function.Parameters,
			},
		})
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// knownTool reports whether name is one of the declared tools. Used to mirror
// parseToolBlocks' Unknown marking for native tool calls.
func knownTool(name string, tools []ToolSpec) bool {
	for _, t := range tools {
		if t.Function.Name == name {
			return true
		}
	}
	return false
}

// nativeToHarnessCalls maps chat-seam native tool calls into the harness ToolCall
// shape, preserving the Unknown flag for names outside the declared surface.
func nativeToHarnessCalls(calls []chat.ToolCall, tools []ToolSpec) []ToolCall {
	if len(calls) == 0 {
		return nil
	}
	out := make([]ToolCall, 0, len(calls))
	for _, c := range calls {
		out = append(out, ToolCall{
			ID:      c.ID,
			Name:    c.Name,
			Args:    c.Arguments,
			Unknown: !knownTool(c.Name, tools),
		})
	}
	return out
}

// parseToolBlocks mirrors Python _parse_tool_calls, for the prompt-based path:
// it never DROPS an unknown call, it marks it Unknown so the tool node replies
// with a correction instead of leaving the model without an answer.
//
// "Unknown" is judged against the FULL ToolMap (Python's _TOOL_MAP), not the
// active per-mode surface. A tool that was runtime-disabled is still a real
// tool — marking it Unknown would make the tool node reply "X is not a tool"
// and never reach ExecuteTool's disabled short-circuit (which returns the
// correct "unavailable in this dataset" note).
func parseToolBlocks(content string, tools []ToolSpec) []ToolCall {
	if content == "" {
		return nil
	}
	_ = tools // active surface is passed by callers but must not drive Unknown
	known := map[string]bool{}
	for name := range ToolMap {
		known[name] = true
	}
	// Prefer the LAST block: when a model emits several, the last is the decision.
	groups := toolBlockRe.FindAllStringSubmatch(content, -1)
	for i := len(groups) - 1; i >= 0; i-- {
		var obj struct {
			Tool string         `json:"tool"`
			Name string         `json:"name"`
			Args map[string]any `json:"args"`
		}
		if err := json.Unmarshal([]byte(groups[i][1]), &obj); err != nil {
			continue
		}
		name := obj.Tool
		if name == "" {
			name = obj.Name
		}
		if name == "" {
			continue
		}
		return []ToolCall{{
			ID:      fmt.Sprintf("call_0"),
			Name:    name,
			Args:    obj.Args,
			Unknown: !known[name],
		}}
	}
	return nil
}

// ---------------------------------------------------------------------------
// Session state
// ---------------------------------------------------------------------------

// SessionState mirrors Python _SessionState (the LangGraph TypedDict).
type SessionState struct {
	// Messages is the running conversation (system + user + assistant + tool).
	Messages []schema.Message
	// ParentState is the slot table being patched by this session.
	ParentState State
	Tools       *Toolset
	Model       SessionModel

	// PendingCalls are tool calls awaiting execution this turn.
	PendingCalls []ToolCall
	Done         bool

	NewStates   []State
	FoundAnswer *string

	RetrievedEvidenceIDs []string
	Attempts             int
	DeadlineLeft         float64

	// CtxBudget is the cumulative tool-payload char ceiling for the session.
	CtxBudget int
	// ToolChars is the running total of tool-payload chars emitted so far
	// (O(1) accounting; Python tracks the same counter as _tool_chars).
	ToolChars int
	// ToolCache avoids re-executing an identical (name, args) call.
	ToolCache map[string]ToolOutcome
	// SearchQueries accumulates retrieval queries for near-dup detection.
	SearchQueries []string
	// SkippedDup counts near-duplicate retrievals suppressed so far.
	SkippedDup int
	// ToolStrikes counts dataset-level empties per tool.
	ToolStrikes map[string]int
	// ToolOutcomes is the audit trail of (name, status, reason, metrics).
	ToolOutcomes []map[string]any

	Direction  string
	RoutedDocs []string
	// NavHint is the joined routed-doc summaries carried by the navigation
	// ladder; it soft-boosts later retrieval ("nav is a hint, not a constraint")
	// without restricting the corpus.
	NavHint string
	// NavRuleID is where the navigation ladder currently rests ("" = the ladder
	// finished or was never armed).
	NavRuleID string

	TerminalType    *string
	TerminalPayload map[string]any
}

// ---------------------------------------------------------------------------
// Tool dispatch
// ---------------------------------------------------------------------------

// ExecuteTool mirrors Python execute_tool: dispatch ONE tool call by name.
//
// Returns a ToolOutcome whose status/reason let the caller act on WHAT happened
// — empty payload, query miss, infra failure, or a run that added no new
// evidence — instead of only counting characters.
func ExecuteTool(ctx context.Context, tools *Toolset, name string, args map[string]any) ToolOutcome {
	// Short-circuit a tool already proven unavailable this session (no compiled
	// structure of its kind). We still return a note, not an error, so the model
	// learns to switch to the corpus tools rather than loop.
	if tools != nil && tools.IsDisabled(name) {
		_LOG.Printf("[Action Session] tool %q disabled (no compiled structure); returning note", name)
		return ToolOutcome{
			Payload: []any{map[string]any{
				"kind": name,
				"note": fmt.Sprintf("%s is unavailable in this dataset (no compiled structure of its kind). Use search_chunks / retrieve / list_chunks instead.", name),
			}},
			Status: StatusEmpty,
			Reason: ReasonNoStructure,
		}
	}
	if tools == nil || tools.Exec == nil {
		return ToolOutcome{Payload: []any{}, Status: StatusError, Reason: ReasonInfra}
	}
	out, err := tools.Exec.Execute(ctx, name, args)
	if err != nil {
		return ToolOutcome{Payload: []any{}, Status: StatusError, Reason: ReasonInfra}
	}
	return out
}

// ---------------------------------------------------------------------------
// Nodes
// ---------------------------------------------------------------------------

// runActionNode mirrors Python _run_action_node: ONE model turn with tools.
// It appends the assistant message and records any tool calls as pending.
func (s *SessionState) runActionNode(ctx context.Context) error {
	s.Attempts++
	// Per-turn wall budget (mirrors Python _run_action_node:1207-1210):
	// max(15, min(75, deadline_left)). Without this a single turn can eat the
	// entire session budget, starving the salvage/finalize steps.
	wall := max(15.0, min(75.0, s.DeadlineLeft))
	callCtx, cancel := context.WithTimeout(ctx, time.Duration(wall*float64(time.Second)))
	defer cancel()
	reply, err := s.Model.Complete(callCtx, s.Messages, s.Tools.ActiveToolSpecs())
	if err != nil {
		return err
	}
	s.Messages = append(s.Messages, *schema.AssistantMessage(reply.Content, nil))

	if len(reply.ToolCalls) > 0 {
		s.PendingCalls = reply.ToolCalls
		return nil
	}
	s.PendingCalls = nil

	// No call: try the terminal protocol (<state> / <answer>).
	newStates, foundAnswer, terminalType, payload := ParseTerminal(reply.Content, s.ParentState)
	if foundAnswer != nil || len(newStates) > 0 {
		s.NewStates = newStates
		s.FoundAnswer = foundAnswer
		s.TerminalType = terminalType
		s.TerminalPayload = payload
		s.Done = true
		return nil
	}
	// Neither a call nor a terminal block: nudge once per turn.
	s.Messages = append(s.Messages, *schema.UserMessage("Call the retrieve tool, or output a <state> patch, or emit <answer>."))
	return nil
}

// toolNode mirrors Python _tool_node: execute pending tool calls, append tool
// responses, and apply the per-outcome policies.
func (s *SessionState) toolNode(ctx context.Context) error {
	if s.Tools == nil {
		return nil
	}
	evidenceIDs := append([]string(nil), s.RetrievedEvidenceIDs...)
	budgetChars := s.CtxBudget
	if budgetChars <= 0 {
		budgetChars = maxToolResponseChars * 4
	}
	used := s.ToolChars
	if s.ToolCache == nil {
		s.ToolCache = map[string]ToolOutcome{}
	}
	seenQueries := append([]string(nil), s.SearchQueries...)
	skipped := 0
	strikes := map[string]int{}
	for k, v := range s.ToolStrikes {
		strikes[k] = v
	}
	// Where the nav ladder currently rests. Bound BEFORE the loop so it survives
	// the empty-pending case and is the input for every tool_call that comes in —
	// a call only advances its own rung, and a later ladder continuation must
	// continue from the SAME resting point.
	//
	// CONTINUATION IS PENDING: the ladder can only be resumed once a rung leaves
	// the chain in ModeLLM (which no rule in NavRules currently does — every
	// rung is ModeAuto), so in practice PendingRule is "" by the time the prefix
	// returns and NavRuleID stays "". The wiring is kept so a future LLM rung
	// resumes from the right place without another change here.
	pendingRule := s.NavRuleID

	for _, c := range s.PendingCalls {
		// Near-duplicate retrieval suppression: if the model re-issues the same
		// intent as an earlier search (paraphrase), do NOT re-run the index —
		// return a nudge so it patches / reframes instead of burning turns.
		q, _ := c.Args["query"].(string)
		q = strings.TrimSpace(q)
		if retrievalTools[c.Name] && q != "" && IsNearDup(q, seenQueries) {
			skipped++
			_LOG.Printf("[Action Session] skipping near-duplicate retrieval %q (already searched)", trunc(q, 80))
			s.Messages = append(s.Messages, toolMessage(c.ID, []any{map[string]any{
				"kind": c.Name,
				"note": "This query is a near-duplicate of an earlier retrieval and was skipped to avoid redundant searching. Patch the slot with what you have, or issue a genuinely NEW retrieval angle.",
			}}))
			continue
		}
		// Unknown tool name — never execute it; answer with a correction so the
		// model can recover. Models usually emit the XML protocol tags (state /
		// answer) as tool names; they belong in the reply body as plain text.
		if c.Unknown {
			hint := fmt.Sprintf(
				"'%s' is not a tool. State patches and final answers are plain TEXT in your reply body, wrapped in <state>...</state> or <answer>...</answer> XML tags — do not emit them as tool calls. Available tools: %s.",
				c.Name, strings.Join(ToolMapNames(), ", "))
			s.Messages = append(s.Messages, toolMessage(c.ID, []any{map[string]any{"kind": "error", "note": hint}}))
			continue
		}

		// Same-session cache: avoid re-running an identical (name, args) call.
		// The cache only avoids RE-EXECUTING; a response is still returned for
		// every declared call, because the assistant message declared it.
		cacheKey := callCacheKey(c)
		oc, cached := s.ToolCache[cacheKey]
		if !cached {
			oc = ExecuteTool(ctx, s.Tools, c.Name, c.Args)
			s.ToolCache[cacheKey] = oc
		}
		if q != "" {
			seenQueries = append(seenQueries, q)
		}
		evidenceIDs = append(evidenceIDs, oc.EvidenceIDs...)
		chunks := append([]any(nil), oc.Payload...)

		// ── Policy: act on WHAT happened, not just on payload size ──────────
		switch {
		case oc.Status == StatusOK:
			// A real hit clears the tool's strike record: it demonstrably works.
			delete(strikes, c.Name)
		case oc.Status == StatusEmpty && oc.Reason == ReasonNoStructure:
			// Dataset-level dead end. Strike it; disable once the strikes pile up
			// so one unlucky scope cannot kill the tool.
			n := strikes[c.Name] + 1
			strikes[c.Name] = n
			if n >= emptyStrikes {
				s.Tools.DisableTool(c.Name)
				_LOG.Printf("[Action Session] %s disabled after %d dataset-level empty results", c.Name, n)
			}
		case oc.Status == StatusRedundant:
			// Ran fine, but every hit was already in the shared evidence pool.
			// Say so explicitly — otherwise the model sees a normal passage list
			// and concludes the search succeeded, then re-searches the same ground.
			chunks = append(chunks, map[string]any{
				"kind": c.Name,
				"note": "All hits from this call were ALREADY in your evidence. Re-reading them will not add anything — issue a genuinely new angle, or output a <state> patch with what you have.",
			})
		}

		s.ToolOutcomes = append(s.ToolOutcomes, map[string]any{
			"name": c.Name, "status": oc.Status, "reason": oc.Reason, "metrics": oc.Metrics,
		})

		payload := marshalPassages(chunks)
		// If the session is already heavy, cut this payload proportionally.
		if used+len(payload) > budgetChars {
			keep := budgetChars - used
			if keep < 800 {
				keep = 800
			}
			payload = payload[:min(len(payload), keep)]
		}
		used += len(payload)
		s.Messages = append(s.Messages, *schema.ToolMessage(payload, c.ID))
	}

	// The ladder keeps advancing: a later tool_call in the SAME batch (or the
	// next turn) resumes from where this call left it, not from the original
	// resting point. "" means the ladder finished — also correct to record.
	// No-op until the ladder continuation is ported (see the note above).
	s.NavRuleID = pendingRule

	s.PendingCalls = nil
	s.RetrievedEvidenceIDs = evidenceIDs
	s.ToolChars = used
	s.SearchQueries = seenQueries
	s.SkippedDup += skipped
	s.ToolStrikes = strikes
	return nil
}

// finalizeNode mirrors Python _finalize_node: tool budget spent — ONE last call
// WITHOUT tools, demanding the terminal JSON to salvage whatever was learned.
func (s *SessionState) finalizeNode(ctx context.Context) error {
	budgetPrompt := ("TOOL BUDGET EXHAUSTED. Based ONLY on the passages retrieved above, output now — no prose outside the block:\n" +
		"<state>{\"new_states\": [{\"state\": [{\"id\": <slot_id>, \"candidate\": \"<value>\", " +
		"\"candidate_strength\": <0..1>, \"discovered_clues\": [\"...\"]}]}]}</state>\n" +
		"If NOTHING was learned use: <state>{\"new_states\": []}</state>")

	// Defensive: strip any assistant.tool_calls that never got a tool response,
	// else the provider rejects the history.
	msgs := append([]schema.Message(nil), s.Messages...)
	msgs = stripUnpairedToolCalls(msgs)
	msgs = append(msgs, *schema.UserMessage(budgetPrompt))

	tmo := s.DeadlineLeft
	if tmo <= 0 {
		tmo = finalizeTimeoutS
	}
	tmo = min(max(tmo, minFinalizeTimeoutS), finalizeTimeoutS)

	callCtx, cancel := context.WithTimeout(ctx, time.Duration(tmo*float64(time.Second)))
	defer cancel()

	reply, err := s.Model.Complete(callCtx, msgs, nil)
	if err != nil {
		_LOG.Printf("[Action Session] finalize call failed: %v", err)
		return nil
	}
	newStates, foundAnswer, terminalType, payload := ParseTerminal(reply.Content, s.ParentState)
	s.NewStates = newStates
	s.FoundAnswer = foundAnswer
	s.TerminalType = terminalType
	s.TerminalPayload = payload

	// Loose-clue harvest (deterministic, zero-LLM; mirrors Python
	// action_session._finalize_node:1497-1518): when the salvage produced no
	// branches AND no answer, the last AI narration often still carries a fact
	// worth keeping as a breadcrumb in the first unresolved slot.
	if len(newStates) == 0 && foundAnswer == nil {
		if txt := lastNarration(s.Messages); len(txt) >= 24 {
			unresolved := s.ParentState.Unresolved()
			var targetID int
			if len(unresolved) > 0 {
				targetID = unresolved[0].ID
			} else if len(s.ParentState.State) > 0 {
				targetID = s.ParentState.State[0].ID
			} else {
				targetID = -1
			}
			if targetID >= 0 {
				if patched := ApplyPatch(s.ParentState, []map[string]any{
					{"id": targetID, "discovered_clues": []string{"narrative: " + txt[:min(len(txt), 220)]}},
				}); patched != nil {
					s.NewStates = []State{*patched}
				}
			}
		}
	}

	s.Done = true
	return nil
}

// lastNarration returns the content of the last assistant message that carries
// non-empty text, scanning newest-first (mirrors Python's reversed(message) ai
// lookup in _finalize_node).
func lastNarration(msgs []schema.Message) string {
	for i := len(msgs) - 1; i >= 0; i-- {
		m := msgs[i]
		if m.Role != schema.Assistant {
			continue
		}
		// Skip assistant messages that only hold tool_calls (no free text).
		if strings.TrimSpace(m.Content) == "" {
			continue
		}
		return strings.TrimSpace(m.Content)
	}
	return ""
}

// ---------------------------------------------------------------------------
// Routing
// ---------------------------------------------------------------------------

type routeTarget int

const (
	routeEnd routeTarget = iota
	routeTool
	routeFinalize
	routeRunAction
)

// actionMaxTurns mirrors Python _action_max_turns: the mode's turn budget.
func (s *SessionState) actionMaxTurns() int {
	spec := ResolveMode(s.Tools)
	if spec.ActionMaxTurns <= 0 {
		return 4
	}
	return spec.ActionMaxTurns
}

// route mirrors Python _route.
func (s *SessionState) route() routeTarget {
	if s.Done {
		return routeEnd
	}
	// Run pending tool_calls FIRST, even at the turn budget: leaving an
	// assistant.tool_calls message without its tool response makes the provider
	// reject the next call. The tool node clears PendingCalls, then the route
	// re-checks the budget.
	if len(s.PendingCalls) > 0 {
		return routeTool
	}
	if s.Attempts >= s.actionMaxTurns() {
		return routeFinalize
	}
	// No-progress convergence: the model keeps re-issuing near-duplicate
	// retrievals, so further turns are unlikely to surface new evidence.
	if s.SkippedDup >= skippedDupLimit {
		_LOG.Printf("[Action Session] %d near-duplicate retrieval(s) skipped; converging session early", s.SkippedDup)
		return routeFinalize
	}
	return routeRunAction
}

// routeAfterTool mirrors Python _route_after_tool: the turn budget is checked
// AFTER the tool responses are appended. A pending tool_call must always
// receive its matching tool response, but once the budget is spent we must NOT
// go back to run_action — that was the Q86 infinite-loop: the model kept
// emitting tool_calls, PendingCalls stayed non-empty, so the attempts check in
// route was never reached and the session burned the whole timeout.
func (s *SessionState) routeAfterTool() routeTarget {
	if s.Attempts >= s.actionMaxTurns() {
		return routeFinalize
	}
	return routeRunAction
}

// sessionLoop drives the three nodes exactly as the compiled LangGraph does:
//
//	START → run_action
//	run_action → route → {END | tool | finalize | run_action}
//	tool → route_after_tool → {run_action | finalize}
//	finalize → END
func (s *SessionState) sessionLoop(ctx context.Context) error {
	node := routeRunAction
	for {
		switch node {
		case routeRunAction:
			if err := s.runActionNode(ctx); err != nil {
				return err
			}
			node = s.route()
		case routeTool:
			if err := s.toolNode(ctx); err != nil {
				return err
			}
			node = s.routeAfterTool()
		case routeFinalize:
			if err := s.finalizeNode(ctx); err != nil {
				return err
			}
			return nil
		case routeEnd:
			return nil
		default:
			return nil
		}
	}
}

// ---------------------------------------------------------------------------
// Terminal parsing
// ---------------------------------------------------------------------------

// ParseTerminal mirrors Python _parse_terminal: parse the two terminal blocks
// (<state> patches → new-state branches; <answer> → final answer).
//
// Returns (newStates, foundAnswer, terminalType, payload) — exactly one of
// newStates / foundAnswer may be non-empty.
func ParseTerminal(content string, parent State) ([]State, *string, *string, map[string]any) {
	if strings.Contains(content, "<state>") {
		block := ExtractTag(content, "state")
		if block == "" {
			block = "{}"
		}
		data, _ := ExtractJSON(block).(map[string]any)
		if data == nil {
			data = map[string]any{}
		}
		rawBranches, _ := data["new_states"].([]any)
		// Tolerate a bare list of variable patches (no {"state": [...]} wrapper):
		// models emit both shapes.
		if len(rawBranches) > 0 {
			allBare := true
			for _, b := range rawBranches {
				if m, ok := b.(map[string]any); ok {
					if _, has := m["state"]; has {
						allBare = false
						break
					}
				}
			}
			if allBare {
				wrapped := make([]any, 0, len(rawBranches))
				for _, b := range rawBranches {
					wrapped = append(wrapped, map[string]any{"state": []any{b}})
				}
				rawBranches = wrapped
			}
		}
		var branches []State
		for _, rb := range rawBranches {
			br, ok := rb.(map[string]any)
			if !ok {
				continue
			}
			patches := toPatchList(br["state"])
			if ns := ApplyPatch(parent, patches); ns != nil {
				branches = append(branches, *ns)
			}
		}
		tt := "state"
		return branches, nil, &tt, data
	}
	if strings.Contains(content, "<answer>") {
		block := ExtractTag(content, "answer")
		if block == "" {
			block = "{}"
		}
		data, _ := ExtractJSON(block).(map[string]any)
		if data == nil {
			data = map[string]any{}
		}
		ans := ""
		if raw, ok := data["answer"]; ok && raw != nil {
			ans = fmt.Sprint(raw)
		}
		patches := toPatchList(data["new_state"])
		var branches []State
		if ns := ApplyPatch(parent, patches); ns != nil {
			branches = append(branches, *ns)
		}
		tt := "answer"
		return branches, &ans, &tt, data
	}
	return nil, nil, nil, nil
}

func toPatchList(v any) []map[string]any {
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
// Helpers
// ---------------------------------------------------------------------------

func toolMessage(id string, passages []any) schema.Message {
	return *schema.ToolMessage(marshalPassages(passages), id)
}

func marshalPassages(passages []any) string {
	if passages == nil {
		passages = []any{}
	}
	raw, err := json.Marshal(map[string]any{"passages": passages})
	if err != nil {
		return `{"passages": []}`
	}
	return string(raw)
}

// callCacheKey mirrors Python's `(name, json.dumps(args, sort_keys=True))` key.
func callCacheKey(c ToolCall) string {
	raw, err := json.Marshal(c.Args)
	if err != nil {
		raw = []byte("{}")
	}
	return c.Name + "|" + string(raw)
}

// stripUnpairedToolCalls removes assistant tool_calls that never received a
// tool response — a malformed history makes providers reject the next request
// ("tool call result does not follow tool call").
func stripUnpairedToolCalls(msgs []schema.Message) []schema.Message {
	answered := map[string]bool{}
	for _, m := range msgs {
		if m.Role == schema.Tool && m.ToolCallID != "" {
			answered[m.ToolCallID] = true
		}
	}
	out := make([]schema.Message, 0, len(msgs))
	for _, m := range msgs {
		if m.Role == schema.Assistant && len(m.ToolCalls) > 0 {
			kept := make([]schema.ToolCall, 0, len(m.ToolCalls))
			for _, tc := range m.ToolCalls {
				if answered[tc.ID] {
					kept = append(kept, tc)
				}
			}
			if len(kept) != len(m.ToolCalls) {
				m.ToolCalls = kept
			}
		}
		out = append(out, m)
	}
	return out
}

func trunc(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n]
}

// sortKeys is a small helper for deterministic map-key iteration in logs.
func sortKeys[V any](m map[string]V) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// ===================== navchain.go =====================
// Deterministic navigation prefix: a code-driven tool chain.
//
// Mirrors Python harness/action_session.py:1614-1933.
//
// WHY THIS EXISTS: the ReAct loop lets the MODEL decide which tool to call
// next, so ordering rules ("route with navigate_tree first, then drill with
// navigate_structure, fall back to retrieve when the drill is weak") can only
// be *suggested* in a prompt and are routinely ignored. This prefix runs those
// rules IN CODE before the loop starts and injects the resulting exchange as
// completed assistant/tool messages: the model then explores on top of a
// guaranteed-correct opening and cannot skip a step. The loop keeps its freedom
// for everything after the chain.

// Step modes.
const (
	// ModeAuto means the orchestrator runs this step itself (no model
	// round-trip).
	ModeAuto = "auto"
	// ModeLLM means the step needs a decision only the model can make (e.g.
	// WHICH routed document to drill into), so the chain stops here and hands
	// control back.
	ModeLLM = "llm"
)

// Navigation-chain constants.
const (
	// navPrefixBudgetRatio is the share of the session deadline the prefix may
	// spend; the ReAct loop keeps the rest.
	navPrefixBudgetRatio = 0.35
	// navPrefixCallTimeoutS is the per-call ceiling, so one slow navigation
	// cannot swallow the whole prefix.
	navPrefixCallTimeoutS = 25.0
	// navHintChars caps the nav-hint text fed back into retrieval as keywords.
	navHintChars = 600
	// navPrefixMinBudgetS floors the prefix budget (Python: max(5.0, ...)).
	navPrefixMinBudgetS = 5.0
	// navMinStepBudgetS floors a single step's budget (Python: max(5.0,
	// remaining)).
	navMinStepBudgetS = 5.0
)

// NavRulesEnabled mirrors Python _NAV_RULES_ENABLED.
const NavRulesEnabled = true

// NavContext is the mutable state threaded through the navigation chain.
//
// KnownDocs is the routed scope (doc_ids). RoutedDocs carries each routed doc's
// OVERALL SUMMARY — the hint that feeds retrieval as a soft boost instead of a
// hard filter. NavHint is the joined summaries used as retrieval keywords so
// routed docs rank up WITHOUT excluding the rest of the corpus.
type NavContext struct {
	Direction  string
	KnownDocs  []string
	RoutedDocs [][2]string // (doc_id, summary)
	NavHint    string
}

// NavRule is one step of the navigation chain.
type NavRule struct {
	ID   string
	Tool string
	// Mode is ModeAuto or ModeLLM.
	Mode string
	// Run replaces the single-tool path for a step that composes several tools
	// (drill: retrieve + navigate_structure + merge).
	Run func(ctx context.Context, ts *Toolset, nav *NavContext, available map[string]bool, budgetS float64) ToolOutcome
	// Args builds the tool arguments from the running context (ModeAuto only;
	// also used for display when Run composes internally).
	Args func(nav *NavContext) map[string]any
	// When is an optional guard; the step is skipped when it returns false.
	When func(nav *NavContext) bool
	// Next maps status -> next rule id. "" (or a missing key) ends the chain.
	// Routing on STATUS is what makes "quality poor -> fall back" a code
	// decision rather than a prompt suggestion.
	Next map[string]string
}

// NavRules is the per-slot strategy ladder.
//
// "Nav is a hint, not a constraint": neither rung filters the corpus down to
// the routed docs — the tree ROUTES (locate), then drill retrieves over the
// WHOLE corpus while softly boosting + re-ranking the routed-doc chunks, so a
// multi-hop / enumeration answer living outside the routed docs survives (it
// just ranks lower) instead of being filtered out.
//
//	locate  navigate_tree    route to top-n documents, exposing their summaries
//	drill   retrieve+merge   whole-corpus retrieve (nav summaries as BM25 soft
//	                         hint) + navigate_structure paths merged on chunk_id
//	                         + routed-doc chunks re-ranked to the top
//	global  retrieve         safety net only if drill itself came back empty
//
// There is deliberately NO LLM verdict on the drill evidence: retrieval is not
// scope-locked, so there is no "did the scoped search miss?" signal to grade
// and no POOR -> global fallback to drive. Sufficiency is the model's + the
// outer SCA's job, not the ladder's.
var NavRules = []NavRule{
	{
		ID:   "locate",
		Tool: "navigate_tree",
		Mode: ModeAuto,
		Args: func(nav *NavContext) map[string]any { return map[string]any{"query": nav.Direction} },
		// Tree missed => no routed hints to merge against; the only useful step
		// is an unscoped search.
		Next: map[string]string{
			StatusOK:        "drill",
			StatusMiss:      "global",
			StatusEmpty:     "global",
			StatusPoor:      "global",
			StatusError:     "global",
			StatusRedundant: "global",
		},
	},
	{
		ID:   "drill",
		Tool: "navigate_structure",
		Mode: ModeAuto,
		Run:  runDrillMerge,
		Args: func(nav *NavContext) map[string]any { return map[string]any{"query": nav.Direction} },
		// drill returns OK with the merged, re-ranked evidence; only an empty
		// whole-corpus result (MISS) or an infra failure falls through to global.
		Next: map[string]string{
			StatusOK:        "",
			StatusMiss:      "global",
			StatusEmpty:     "global",
			StatusError:     "global",
			StatusRedundant: "global",
		},
	},
	{
		ID:   "global",
		Tool: "retrieve",
		Mode: ModeAuto,
		Args: func(nav *NavContext) map[string]any { return map[string]any{"query": []string{nav.Direction}} },
		Next: map[string]string{},
	},
}

var navRuleByID = func() map[string]*NavRule {
	m := make(map[string]*NavRule, len(NavRules))
	for i := range NavRules {
		m[NavRules[i].ID] = &NavRules[i]
	}
	return m
}()

// NavStartRule is the ladder's entry point.
const NavStartRule = "locate"

// runDrillMerge is the `drill` rung (AUTO): corpus retrieve for REAL chunks +
// the root->chunk structure paths from navigate_structure, MERGED on chunk_id
// so every retrieved chunk carries its hierarchy context.
//
// Nav is a hint, not a constraint: retrieval runs over the WHOLE corpus (no
// doc_scope filter), but the nav-routed docs' summaries are fed back as
// retrieval keywords (soft boost), and the returned chunks are re-ranked so
// routed-doc chunks float to the top while non-routed chunks stay below. A
// multi-hop / enumeration answer living OUTSIDE the routed docs therefore
// survives — it just ranks lower — instead of being filtered out entirely.
func runDrillMerge(ctx context.Context, ts *Toolset, nav *NavContext, available map[string]bool, budgetS float64) ToolOutcome {
	var merged []any
	var evidenceIDs []string
	seenIDs := map[string]bool{}

	// 1. Skeleton A: whole-corpus retrieval, softly boosted by the nav hints.
	retOC := ExecuteTool(ctx, ts, "retrieve", map[string]any{"query": []string{nav.Direction}})
	for _, cid := range retOC.EvidenceIDs {
		if !seenIDs[cid] {
			seenIDs[cid] = true
			evidenceIDs = append(evidenceIDs, cid)
		}
	}
	routedIDs := map[string]bool{}
	for _, d := range nav.KnownDocs {
		routedIDs[d] = true
	}

	// 2. Supplement B: root->chunk paths from EVERY routed doc's structure.
	paths := map[string]string{}
	if available["navigate_structure"] && len(nav.KnownDocs) > 0 {
		stepCtx, cancel := context.WithTimeout(ctx, stepBudget(budgetS))
		defer cancel()
		for _, docID := range nav.KnownDocs {
			select {
			case <-stepCtx.Done():
				break
			default:
			}
			sOC := ExecuteTool(stepCtx, ts, "navigate_structure", map[string]any{
				"doc_id": docID, "query": nav.Direction, "kind": "catalog",
			})
			for _, d := range sOC.EvidenceIDs {
				if !seenIDs[d] {
					seenIDs[d] = true
					evidenceIDs = append(evidenceIDs, d)
				}
			}
			for cid, path := range chunkPathsFromMetrics(sOC.Metrics) {
				if cid != "" {
					if _, ok := paths[cid]; !ok {
						paths[cid] = path
					}
				}
			}
		}
	}

	// 3. Merge: skeleton A is the base; attach structure_path where ids align.
	//    Re-rank so routed-doc chunks float to the top WITHOUT deleting the rest.
	for _, p := range retOC.Payload {
		e, ok := p.(map[string]any)
		if !ok {
			merged = append(merged, map[string]any{"content": fmt.Sprint(p)})
			continue
		}
		entry := make(map[string]any, len(e)+2)
		for k, v := range e {
			entry[k] = v
		}
		if cid, ok := entry["id"].(string); ok && cid != "" {
			if sp, ok := paths[cid]; ok {
				entry["structure_path"] = sp
			}
		}
		doc := ""
		if v, ok := entry["doc_id"].(string); ok {
			doc = v
		} else if v, ok := entry["document_id"].(string); ok {
			doc = v
		}
		rank := 1
		if routedIDs[doc] {
			rank = 0 // 0 = routed (top)
		}
		entry["nav_rank"] = rank
		merged = append(merged, entry)
	}
	// Stable sort: routed first, original order preserved within each group
	// (Go's sort.Slice is not stable).
	sort.SliceStable(merged, func(i, j int) bool {
		return navRankOf(merged[i]) < navRankOf(merged[j])
	})

	if len(merged) == 0 {
		return ToolOutcome{
			Payload:     []any{},
			EvidenceIDs: evidenceIDs,
			Status:      StatusMiss,
			Reason:      ReasonNoDoc,
			Metrics:     map[string]any{"hits": 0},
		}
	}
	return ToolOutcome{
		Payload:     merged,
		EvidenceIDs: evidenceIDs,
		Status:      StatusOK,
		Metrics:     map[string]any{"hits": len(merged), "routed": len(routedIDs)},
	}
}

func navRankOf(v any) int {
	m, ok := v.(map[string]any)
	if !ok {
		return 1
	}
	if n, ok := m["nav_rank"].(int); ok {
		return n
	}
	return 1
}

// stepBudget applies the per-step floor (Python: max(5.0, remaining)).
func stepBudget(remaining float64) time.Duration {
	if remaining < navMinStepBudgetS {
		remaining = navMinStepBudgetS
	}
	return time.Duration(remaining * float64(time.Second))
}

// chunkPathsFromMetrics reads metrics["chunk_paths"], tolerating the shape drift
// of model/executor-produced metrics.
func chunkPathsFromMetrics(metrics map[string]any) map[string]string {
	out := map[string]string{}
	raw, ok := metrics["chunk_paths"]
	if !ok || raw == nil {
		return out
	}
	switch v := raw.(type) {
	case map[string]string:
		return v
	case map[string]any:
		for k, val := range v {
			out[k] = fmt.Sprint(val)
		}
	}
	return out
}

// navToolSurface is the set of tools this session may call.
func (t *Toolset) navToolSurface() map[string]bool {
	out := map[string]bool{}
	for _, s := range t.ActiveToolSpecs() {
		out[s.Function.Name] = true
	}
	return out
}

// NavExchange is one completed assistant/tool pair produced by the chain.
type NavExchange struct {
	Messages    []schema.Message
	EvidenceIDs []string
	Outcomes    []map[string]any
	// PendingRule is the rule control now rests on ("" when the ladder finished
	// or was abandoned).
	PendingRule string
	// BudgetLeft is the unspent prefix budget in seconds.
	BudgetLeft float64
}

// RunNavChain mirrors Python _run_nav_chain: run NavRules from startID, stopping
// at the first LLM step.
//
// AUTO steps execute here; an LLM step is returned without being run, because
// only the model can supply its arguments (e.g. which routed document to drill).
// Shared by the pre-session prefix and the in-session fallback, so both consume
// exactly the same ladder.
func RunNavChain(ctx context.Context, ts *Toolset, nav *NavContext, startID string, budgetS float64) NavExchange {
	var out NavExchange
	if !NavRulesEnabled {
		return out
	}
	available := ts.navToolSurface()

	started := time.Now()
	ruleID := startID
	for ruleID != "" {
		rule, ok := navRuleByID[ruleID]
		if !ok || !available[rule.Tool] {
			break
		}
		if rule.Mode == ModeLLM {
			// Hand control back: the model must decide.
			out.PendingRule = ruleID
			out.BudgetLeft = budgetS - time.Since(started).Seconds()
			return out
		}
		if rule.When != nil && !rule.When(nav) {
			break
		}
		remaining := budgetS - time.Since(started).Seconds()
		if remaining <= 1.0 {
			_LOG.Printf("[Action Session] nav chain out of budget before %q", ruleID)
			break
		}

		var oc ToolOutcome
		if rule.Run != nil {
			// A composed step (e.g. drill) owns its own tool calls.
			oc = rule.Run(ctx, ts, nav, available, remaining)
		} else {
			stepCtx, cancel := context.WithTimeout(ctx, minDuration(stepBudget(remaining),
				time.Duration(navPrefixCallTimeoutS*float64(time.Second))))
			oc = ExecuteTool(stepCtx, ts, rule.Tool, rule.Args(nav))
			cancel()
		}

		out.Outcomes = append(out.Outcomes, map[string]any{
			"name": rule.Tool, "status": oc.Status, "reason": oc.Reason, "metrics": oc.Metrics,
		})
		if rule.Tool == "navigate_tree" && oc.Status == StatusOK {
			// Routed docs become the validated scope for every later step.
			nav.KnownDocs = appendUnique(nav.KnownDocs, docIDsFromPayload(oc.Payload)...)
			nav.RoutedDocs = routedDocsFromMetrics(oc.Metrics)
			var hints []string
			for _, rd := range nav.RoutedDocs {
				if rd[1] != "" {
					hints = append(hints, rd[1])
				}
			}
			if len(hints) > 0 {
				nav.NavHint = truncateRunes(strings.Join(hints, " "), navHintChars)
			}
		}
		out.EvidenceIDs = append(out.EvidenceIDs, oc.EvidenceIDs...)
		out.consumeExchange(ruleID, rule.Tool, rule.Args(nav), oc)
		ruleID = rule.Next[oc.Status]
	}
	out.PendingRule = ruleID
	out.BudgetLeft = budgetS - time.Since(started).Seconds()
	return out
}

// consumeExchange appends one completed assistant/tool exchange.
//
// The pair MUST be well formed — every tool_call needs its tool response, or
// the provider rejects the history (the reason stripUnpairedToolCalls exists).
func (e *NavExchange) consumeExchange(ruleID, tool string, args map[string]any, oc ToolOutcome) {
	callID := fmt.Sprintf("nav_%s", ruleID)
	rawArgs, err := json.Marshal(args)
	if err != nil {
		rawArgs = []byte("{}")
	}
	e.Messages = append(e.Messages,
		*schema.AssistantMessage("", []schema.ToolCall{{
			ID:   callID,
			Type: "function",
			Function: schema.FunctionCall{
				Name:      tool,
				Arguments: string(rawArgs),
			},
		}}),
		*schema.ToolMessage(marshalPassages(oc.Payload), callID),
	)
}

// RunNavPrefix mirrors Python run_nav_prefix: run the navigation ladder up to
// the first model-driven step.
//
// The returned messages are completed assistant/tool pairs ready to seed the
// session history, so the model sees the chain as work already done and cannot
// skip a step.
//
// Returns an empty exchange when navigation is unavailable — the ladder is an
// optimisation, never a precondition for the session to run.
func RunNavPrefix(ctx context.Context, ts *Toolset, direction string, deadlineLeft float64, nav *NavContext) NavExchange {
	if !NavRulesEnabled || ts == nil {
		return NavExchange{}
	}
	available := ts.navToolSurface()
	if len(available) == 0 || !available["navigate_tree"] {
		return NavExchange{}
	}
	if nav == nil {
		nav = &NavContext{Direction: direction}
	}
	budget := deadlineLeft * navPrefixBudgetRatio
	if budget < navPrefixMinBudgetS {
		budget = navPrefixMinBudgetS
	}
	return RunNavChain(ctx, ts, nav, NavStartRule, budget)
}

// ---------------------------------------------------------------------------
// Payload helpers
// ---------------------------------------------------------------------------

// docIDsFromPayload reads payload[0]["doc_ids"] from a navigate_tree result.
func docIDsFromPayload(payload []any) []string {
	if len(payload) == 0 {
		return nil
	}
	first, ok := payload[0].(map[string]any)
	if !ok {
		return nil
	}
	return toStringList(first["doc_ids"])
}

// routedDocsFromMetrics reads metrics["routed_docs"] — a list of (doc_id,
// summary) pairs, which models/executors emit either as [doc, summary] lists or
// as {"doc_id":..., "summary":...} objects.
func routedDocsFromMetrics(metrics map[string]any) [][2]string {
	raw, ok := metrics["routed_docs"]
	if !ok || raw == nil {
		return nil
	}
	list, ok := raw.([]any)
	if !ok {
		return nil
	}
	out := make([][2]string, 0, len(list))
	for _, item := range list {
		switch v := item.(type) {
		case []any:
			if len(v) >= 2 {
				out = append(out, [2]string{fmt.Sprint(v[0]), fmt.Sprint(v[1])})
			} else if len(v) == 1 {
				out = append(out, [2]string{fmt.Sprint(v[0]), ""})
			}
		case []string:
			if len(v) >= 2 {
				out = append(out, [2]string{v[0], v[1]})
			} else if len(v) == 1 {
				out = append(out, [2]string{v[0], ""})
			}
		case map[string]any:
			id := fmt.Sprint(v["doc_id"])
			summary := ""
			if s, ok := v["summary"].(string); ok {
				summary = s
			}
			if id != "" {
				out = append(out, [2]string{id, summary})
			}
		}
	}
	return out
}

func toStringList(v any) []string {
	list, ok := v.([]any)
	if !ok {
		if ss, ok := v.([]string); ok {
			return ss
		}
		return nil
	}
	out := make([]string, 0, len(list))
	for _, item := range list {
		if item == nil {
			continue
		}
		if s := fmt.Sprint(item); s != "" {
			out = append(out, s)
		}
	}
	return out
}

func appendUnique(existing []string, add ...string) []string {
	seen := make(map[string]bool, len(existing))
	for _, v := range existing {
		seen[v] = true
	}
	for _, v := range add {
		if !seen[v] {
			seen[v] = true
			existing = append(existing, v)
		}
	}
	return existing
}

func minDuration(a, b time.Duration) time.Duration {
	if a < b {
		return a
	}
	return b
}

// ===================== run.go (entry points, LAST to mirror Python) =====================
// The two public entry points of the action session — placed at the END of the
// file, mirroring Python action_session.py::run_action_session (line 1967) and
// initialize_state (line 2080) being the final definitions.
//
// Mirrors Python harness/action_session.py:
//   - run_action_session (line 1967)
//   - initialize_state   (line 2080)

// PromptLoader loads a named prompt template (e.g. "action_run",
// "action_initialize_state"). Python resolves these from rag/prompts/*.md via
// load_prompt; the Go harness takes the seam as an interface so the templates
// can come from the same files, from embedded assets, or from a test double.
type PromptLoader interface {
	Load(name string) (string, error)
}

// StringPromptLoader serves templates from an in-memory map. Used by tests and
// as the fallback when no filesystem loader is wired.
type StringPromptLoader map[string]string

// Load implements PromptLoader.
func (p StringPromptLoader) Load(name string) (string, error) {
	if t, ok := p[name]; ok {
		return t, nil
	}
	return "", fmt.Errorf("harness: prompt %q not found", name)
}

// SessionDeps are the dependencies of one action session.
type SessionDeps struct {
	Tools *Toolset
	Model SessionModel
	// Prompts supplies the "action_run" / "action_initialize_state" templates.
	// When nil, built-in fallback instructions are used.
	Prompts PromptLoader
	// KB is the shared evidence pool accumulated across the research so far.
	// When non-nil, its chunks are injected into the seed user prompt as
	// "ALREADY RETRIEVED" so the model does not re-retrieve evidence it already
	// has (mirrors Python run_action_session:1982-1984). Nil skips the injection.
	KB *Kbinfos
}

// RunActionSession mirrors Python run_action_session: a bounded session
// pursuing ONE direction.
//
// Returns an empty Result (no states) when no model is configured or the
// session fails — mirroring Python, which logs and returns
// Result(messages=[], new_states=[]) rather than propagating.
func RunActionSession(ctx context.Context, deps SessionDeps, direction string, parent State, deadlineLeft float64, baseSummary string, sharedToolCache map[string]ToolOutcome, sharedSearchQueries []string) Result {
	system := loadPrompt(deps.Prompts, "action_run")
	seedUser := fmt.Sprintf("Direction: %s\n\nState:\n%s", direction, parent.RenderSlots())

	// ALREADY RETRIEVED (mirrors Python run_action_session:1982-1984):
	// surface the evidence already in the shared pool so the model fills slots
	// from it instead of re-retrieving the same ground. Without this the ReAct
	// loop repeatedly searches evidence it already holds.
	if existing := extractRelevantEvidence(deps.KB, direction, 4); existing != "" {
		seedUser += "\n\nALREADY RETRIEVED (do NOT re-retrieve these — use them to fill slots or identify gaps):\n" + existing
	}

	if len(baseSummary) > 0 {
		seedUser += fmt.Sprintf("\n\nPrior round summary:\n%s", baseSummary)
	}

	if deps.Model == nil {
		_LOG.Printf("[Action Session] no usable model resolved for action session")
		return Result{Messages: nil, NewStates: nil}
	}

	budgetLeft := deadlineLeft
	if budgetLeft <= 0 {
		budgetLeft = actionTimeoutS
	}

	st := &SessionState{
		Messages: []schema.Message{
			*schema.SystemMessage(system),
			*schema.UserMessage(seedUser),
		},
		ParentState:          parent,
		Tools:                deps.Tools,
		Model:                deps.Model,
		PendingCalls:         nil,
		Done:                 false,
		NewStates:            nil,
		FoundAnswer:          nil,
		RetrievedEvidenceIDs: nil,
		Attempts:             0,
		DeadlineLeft:         budgetLeft,
		CtxBudget:            maxToolResponseChars * 4,
		ToolChars:            0,
		ToolCache:            sharedToolCache,
		SearchQueries:        sharedSearchQueries,
		SkippedDup:           0,
		ToolStrikes:          map[string]int{},
		ToolOutcomes:         nil,
		Direction:            direction,
	}
	if st.ToolCache == nil {
		st.ToolCache = map[string]ToolOutcome{}
	}

	// The graph loop is bounded by the turn budget and the deadline; the context
	// carries the wall-clock so a stalled provider cannot outlive the request.
	runCtx, cancel := context.WithTimeout(ctx, deadlineToDuration(budgetLeft))
	defer cancel()

	// Deterministic navigation prefix: run the ladder IN CODE and seed the
	// conversation with the completed exchanges, so the model starts on top of a
	// guaranteed-correct opening instead of being merely *told* the routing rule.
	// The prefix shares the ladder with the in-session fallback, and its pending
	// rule becomes the session's resting point (see SessionState.NavRuleID).
	nav := &NavContext{Direction: direction}
	prefix := RunNavPrefix(runCtx, deps.Tools, direction, budgetLeft, nav)
	if len(prefix.Messages) > 0 {
		st.Messages = append(st.Messages, prefix.Messages...)
		st.RetrievedEvidenceIDs = append(st.RetrievedEvidenceIDs, prefix.EvidenceIDs...)
		st.ToolOutcomes = append(st.ToolOutcomes, prefix.Outcomes...)
		st.NavRuleID = prefix.PendingRule
		st.RoutedDocs = nav.KnownDocs
		st.NavHint = nav.NavHint
		_LOG.Printf("[Action Session] nav prefix: %d exchange(s), %d evidence id(s), resting on %q",
			len(prefix.Messages)/2, len(prefix.EvidenceIDs), prefix.PendingRule)
	}

	if err := st.sessionLoop(runCtx); err != nil {
		_LOG.Printf("[Action Session] session failed: %v", err)
		return Result{Messages: nil, NewStates: nil}
	}
	return Result{
		Messages:             st.Messages,
		NewStates:            st.NewStates,
		FoundAnswer:          st.FoundAnswer,
		RetrievedEvidenceIDs: st.RetrievedEvidenceIDs,
		TerminalType:         st.TerminalType,
		TerminalPayload:      st.TerminalPayload,
	}
}

// InitResult mirrors Python initialize_state's tuple return: the root slot
// table plus the queries for the first round.
type InitResult struct {
	Root         State
	FirstQueries []string
}

// InitializeState mirrors Python initialize_state: ask the model to decompose
// the question into a slot table, with a deterministic fallback when the call
// fails.
//
// deadlineLeft bounds the decomposition call (Python: min(_INIT_TIMEOUT_S,
// deadline_left or _INIT_TIMEOUT_S)).
func InitializeState(ctx context.Context, deps SessionDeps, question string, fanoutHint []string, deadlineLeft float64) InitResult {
	system := loadPrompt(deps.Prompts, "action_initialize_state")
	user := "Question: " + question
	if len(fanoutHint) > 0 {
		user += "\n\nCandidate aspects already identified:\n"
		for _, h := range fanoutHint {
			user += "- " + h + "\n"
		}
	}
	tmo := initTimeoutS
	if deadlineLeft > 0 && deadlineLeft < tmo {
		tmo = deadlineLeft
	}

	raw := initChat(ctx, deps, system, user, tmo)
	data, _ := ExtractJSON(raw).(map[string]any)
	if len(data) == 0 {
		// One quick retry — transient provider stalls were observed (45s with
		// zero bytes); a second attempt succeeded in production logs.
		raw = initChat(ctx, deps, system, user, tmo)
		data, _ = ExtractJSON(raw).(map[string]any)
	}

	var slots []Variable
	if rawList, ok := data["slots"].([]any); ok {
		for i, s := range rawList {
			m, ok := s.(map[string]any)
			if !ok {
				continue
			}
			id := i
			if v, ok := toInt(m["id"]); ok {
				id = v
			}
			vType := "entity"
			if v, ok := m["type"].(string); ok && v != "" {
				vType = v
			}
			var clues []string
			if rawClues, ok := m["clues"].([]any); ok {
				for j, c := range rawClues {
					if j >= 4 {
						break
					}
					clues = append(clues, fmt.Sprint(c))
				}
			}
			slots = append(slots, Variable{ID: id, Type: vType, QuestionClues: clues})
		}
	}
	var firstQueries []string
	if rawQ, ok := data["first_queries"].([]any); ok {
		for j, q := range rawQ {
			if j >= 3 {
				break
			}
			if s := strings.TrimSpace(fmt.Sprint(q)); s != "" {
				firstQueries = append(firstQueries, s)
			}
		}
	}

	if len(slots) == 0 {
		// Decomposition failed (timeout/parse): build the table from planner
		// fanouts so the first round still targets DISTINCT aspects instead of
		// one oversized query (observed: single-slot trees answered directly and
		// died, or grepped the raw question and missed).
		if len(fanoutHint) > 0 {
			limit := min(len(fanoutHint), 4)
			for i := 0; i < limit; i++ {
				h := fanoutHint[i]
				if len([]rune(h)) > 120 {
					h = string([]rune(h)[:120])
				}
				slots = append(slots, Variable{ID: i, Type: "aspect", QuestionClues: []string{h}})
			}
			if len(firstQueries) == 0 {
				limit = min(len(fanoutHint), 3)
				for i := 0; i < limit; i++ {
					firstQueries = append(firstQueries, fanoutHint[i])
				}
			}
		} else {
			slots = []Variable{{ID: 0, Type: "answer", QuestionClues: []string{question}}}
			if len(firstQueries) == 0 {
				firstQueries = []string{question}
			}
		}
	} else if len(firstQueries) == 0 {
		firstQueries = []string{question}
	}

	root := NewState(slots, 0, nil)
	_LOG.Printf("[Action Session:init] %s\n%s", root.Brief(), root.RenderSlots())
	return InitResult{Root: root, FirstQueries: firstQueries}
}

// initChat mirrors Python _init_chat: ONE bounded LLM turn for the slot-table
// decomposition. Returns "" on timeout or failure — the caller falls back.
func initChat(ctx context.Context, deps SessionDeps, system, user string, tmo float64) string {
	if deps.Model == nil {
		return ""
	}
	callCtx, cancel := context.WithTimeout(ctx, deadlineToDuration(tmo))
	defer cancel()
	reply, err := deps.Model.Complete(callCtx, []schema.Message{
		*schema.SystemMessage(system),
		*schema.UserMessage(user),
	}, nil)
	if err != nil {
		_LOG.Printf("[Action Session:init] failed: %v", err)
		return ""
	}
	return reply.Content
}

// resolveLoader returns the configured PromptLoader, or the embedded
// Markdown-backed loader when none is wired. Mirrors Python, where
// rag/prompts/template.load_prompt always reads the .md files from
// rag/prompts/ — the harness defaults to the same authoritative templates
// instead of the terse Go fallback constants.
// ---------------------------------------------------------------------------
// Model / tool-calling seam (mirrors Python action_session.py's tool specs +
// _parse_tool_calls output + _llm_once_with_tools → _acompletion).
// ---------------------------------------------------------------------------

// ToolFunction is the OpenAI-style function descriptor.
type ToolFunction struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters"`
}

// ToolSpec is an OpenAI-style tool schema, byte-compatible with the Python
// dicts so a provider accepting either sees the same surface.
type ToolSpec struct {
	Type     string       `json:"type"`
	Function ToolFunction `json:"function"`
}

// ToolCall is one tool invocation requested by the model.
type ToolCall struct {
	ID      string
	Name    string
	Args    map[string]any
	Unknown bool // name is not in ToolMap — answer with a correction, never execute
}

// ModelReply is one model turn: either free text, or tool calls, or both.
type ModelReply struct {
	Content   string
	ToolCalls []ToolCall
	Tokens    int // total tokens, for convenience (== Usage.TotalTokens when Usage != nil)
	// Usage is the provider-reported prompt/completion/total split. Nil when the
	// provider reported nothing.
	Usage *chat.Usage
}

// TemperatureModel is an OPTIONAL extension of SessionModel: models that can
// vary the sampling temperature per call implement it.
//
// Python sets a per-node temperature (keyword extraction uses 0.1, the answer
// composition uses the default). The Go seam originally had no temperature at
// all, so every node ran at whatever the invoker defaulted to. Callers that
// need a specific temperature type-assert to this interface and fall back to
// Complete when it is not implemented.
type TemperatureModel interface {
	CompleteWithTemperature(ctx context.Context, messages []schema.Message, tools []ToolSpec, temp float64) (*ModelReply, error)
}

// ContextLengthModel is implemented by models that can report their context
// window in tokens (Python LLMBundle.max_length), which message-fitting nodes
// use as the budget for chat.FitMessages. Callers type-assert for it and fall
// back to chat.EffectiveContextLength's 8192 default when it is absent, exactly
// mirroring Python's LLM.max_length defaulting when the model config omits
// max_tokens. This is the message_fit_in counterpart to the TemperatureModel
// seam.
type ContextLengthModel interface {
	ContextLength() int
}

// SessionModel is ONE model turn with THIS mode's tool surface.
//
// Python calls this via _llm_once_with_tools → _acompletion. The Go chat seam
// has no native tools field, so the default implementation (see
// InvokerSessionModel) asks for the call as a JSON block. Swap in a native
// implementation when the chat seam grows a tools field.
type SessionModel interface {
	Complete(ctx context.Context, messages []schema.Message, tools []ToolSpec) (*ModelReply, error)
}

// StreamingSessionModel is implemented by models that can emit the answer
// incrementally. Callers type-assert for it and fall back to the one-shot
// Complete when it is absent, so streaming is strictly an enhancement (Python
// tools.answer_sink, fed by the graph's token stream).
type StreamingSessionModel interface {
	SessionModel
	// StreamComplete sends the reply in pieces. onDelta receives each piece and
	// isThink tells whether it belongs to a hidden reasoning block; both are
	// forwarded, matching Python's answer_sink(delta, kind == "think").
	StreamComplete(ctx context.Context, messages []schema.Message, tools []ToolSpec, onDelta func(delta string, isThink bool) error) (*ModelReply, error)
}

// Parsing of model output.
//
// Every LLM call in the harness asks for JSON, and models wrap that JSON in
// thinking preamble and Markdown fences. Python leans on json_repair for
// leniency; the Go equivalent is to strip the wrappers first and let
// encoding/json handle the rest. This mirrors Python's extract_json
// (action_session.py:512), kept in the same file for parity.

var reFence = regexp.MustCompile("```(?:json)?\\s*|\\s*```")

// UnmarshalModelJSON mirrors Python's extract_json: strip any thinking
// preamble and Markdown fences, then parse JSON. An empty result parses as an
// empty object so callers can index into `out` without a nil check.
func UnmarshalModelJSON(text string, out any) error {
	text = common.StripThinkTrailing(text)
	text = reFence.ReplaceAllString(text, "")
	text = strings.TrimSpace(text)
	if text == "" {
		return json.Unmarshal([]byte("{}"), out)
	}
	return json.Unmarshal([]byte(text), out)
}

func resolveLoader(p PromptLoader) PromptLoader {
	if p != nil {
		return p
	}
	return prompts.EmbeddedPromptLoader{}
}

// loadPrompt resolves the prompt loader (defaulting to prompts.EmbeddedPromptLoader
// when none is wired on SessionDeps) and loads name. The canonical templates are
// the Markdown files action_run.md and action_initialize_state.md under
// internal/rag/prompts (a copy of rag/prompts/*.md on the Python side, embedded
// into the binary via //go:embed so they are never absent at runtime). This
// mirrors Python's rag/prompts/template.py::load_prompt: a missing template is an
// error, not a silent fallback to a stale string.
func loadPrompt(p PromptLoader, name string) string {
	t, err := resolveLoader(p).Load(name)
	if err != nil {
		panic(fmt.Sprintf("loadPrompt(%q): %v", name, err))
	}
	return t
}

// deadlineToDuration converts a seconds budget to a context deadline.
// A non-positive budget falls back to the default session timeout so a caller
// that omits it does not produce an already-expired context.
func deadlineToDuration(seconds float64) time.Duration {
	if seconds <= 0 {
		seconds = actionTimeoutS
	}
	return time.Duration(seconds * float64(time.Second))
}

// extractRelevantEvidence mirrors Python action_session._extract_relevant_evidence
// (action_session.py:1939 — placed here, just before the entry points, to match
// Python's source order): flatten the shared evidence pool into a compact,
// line-delimited digest the model can read without re-retrieving. Chunks are
// ranked by relevance to the direction tokens (mirroring Python's token-match
// count sort), then the top maxChunks are surfaced, each truncated at 300 chars
// so the seed prompt stays bounded.
func extractRelevantEvidence(kb *Kbinfos, direction string, maxChunks int) string {
	if kb == nil || maxChunks <= 0 {
		return ""
	}
	chunks := kb.Chunks
	if len(chunks) == 0 {
		return ""
	}

	// Build the set of direction tokens (>=2 chars, alnum or CJK) exactly as
	// Python does, then rank chunks by how many tokens appear in their text.
	dirTokens := tokenizeDirection(direction)
	var ranked []map[string]any
	if len(dirTokens) == 0 {
		// No usable direction: mirror Python's chunks[-max_chunks:] fallback.
		if len(chunks) > maxChunks {
			ranked = chunks[len(chunks)-maxChunks:]
		} else {
			ranked = chunks
		}
	} else {
		type scored struct {
			chunk map[string]any
			score int
		}
		scoredChunks := make([]scored, 0, len(chunks))
		for _, c := range chunks {
			scoredChunks = append(scoredChunks, scored{chunk: c, score: directionRelevance(c, dirTokens)})
		}
		sort.SliceStable(scoredChunks, func(i, j int) bool {
			return scoredChunks[i].score > scoredChunks[j].score
		})
		if len(scoredChunks) > maxChunks {
			scoredChunks = scoredChunks[:maxChunks]
		}
		for _, s := range scoredChunks {
			ranked = append(ranked, s.chunk)
		}
	}

	var b strings.Builder
	for _, c := range ranked {
		content := strings.TrimSpace(ChunkTextOf(c))
		if content == "" {
			continue
		}
		// Mirror Python: hard-cut at 300 (no ellipsis) and flatten newlines so
		// the digest stays single-line per chunk and matches Python output.
		if len(content) > 300 {
			content = content[:300]
		}
		content = strings.ReplaceAll(content, "\n", " ")
		// Mirror Python f"[{cid}] {text}" so the model can cite the chunk id.
		b.WriteString("[")
		b.WriteString(ChunkIDOf(c))
		b.WriteString("] ")
		b.WriteString(content)
		b.WriteString("\n")
	}
	return strings.TrimRight(b.String(), "\n")
}

// tokenizeDirection splits a direction string into the same >=2-char alnum/CJK
// tokens Python's re.findall(r"[a-zA-Z0-9\u4e00-\u9fff]{2,}") produces.
func tokenizeDirection(direction string) []string {
	if direction == "" {
		return nil
	}
	seen := make(map[string]struct{})
	var tokens []string
	var buf strings.Builder
	flush := func() {
		if buf.Len() >= 2 {
			tok := buf.String()
			if _, dup := seen[tok]; !dup {
				seen[tok] = struct{}{}
				tokens = append(tokens, tok)
			}
		}
		buf.Reset()
	}
	for _, r := range strings.ToLower(direction) {
		switch {
		case (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') || (r >= '一' && r <= '鿿'):
			buf.WriteRune(r)
		default:
			flush()
		}
	}
	flush()
	return tokens
}

// directionRelevance counts how many direction tokens appear in a chunk's text,
// mirroring Python's _rel() token-match sum.
func directionRelevance(chunk map[string]any, dirTokens []string) int {
	text := strings.ToLower(ChunkTextOf(chunk))
	if text == "" {
		return 0
	}
	score := 0
	for _, t := range dirTokens {
		if strings.Contains(text, t) {
			score++
		}
	}
	return score
}
