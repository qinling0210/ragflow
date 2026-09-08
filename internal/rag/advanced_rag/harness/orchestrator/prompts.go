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
	"regexp"
	"strings"

	"ragflow/internal/rag/advanced_rag/harness"
	"ragflow/internal/rag/prompts"
)

// Prompt rendering and the built-in fallbacks.
//
// The canonical templates live in the .md files embedded by the
// internal/rag/prompts package (sca_select.md and sca_query_rewrite.md, 1:1
// copies of the Python rag/prompts/*.md). An unset loader defaults to the
// embedded ones, so a wired harness reads the same authoritative text as
// Python. The constants below are used only when the loader cannot resolve the
// name, so a stage degrades to a working protocol rather than failing.

// leftoverVarRE matches a Jinja variable placeholder that no provided var
// filled in. Python renders these through PROMPT_JINJA_ENV's default Undefined
// (SandboxedEnvironment without undefined=StrictUndefined), which interpolates
// as the empty string; the regex mirrors that instead of leaving {{ var }}
// verbatim in the prompt.
var leftoverVarRE = regexp.MustCompile(`\{\{\s*[A-Za-z_][A-Za-z0-9_.]*\s*\}\}`)

// renderPrompt loads `name` and substitutes the {{var}} placeholders the way
// PROMPT_JINJA_ENV.from_string(...).render(...) does on the Python side. An
// unset loader falls back to the embedded canonical .md templates; a loader
// that cannot resolve `name` falls back to `fallback`.
func renderPrompt(loader harness.PromptLoader, name, fallback string, vars map[string]string) string {
	if loader == nil {
		loader = prompts.EmbeddedPromptLoader{}
	}
	tmpl := fallback
	if t, err := loader.Load(name); err == nil && strings.TrimSpace(t) != "" {
		tmpl = t
	}
	for k, v := range vars {
		// Jinja tolerates both `{{var}}` and `{{ var }}`; the project's
		// templates use the spaced form, so substitute both.
		tmpl = strings.ReplaceAll(tmpl, "{{"+k+"}}", v)
		tmpl = strings.ReplaceAll(tmpl, "{{ "+k+" }}", v)
	}
	// Unprovided variables render as the empty string (Jinja default
	// Undefined), not as a literal placeholder.
	return leftoverVarRE.ReplaceAllString(tmpl, "")
}

const fallbackSCAReview = `You are a research auditor reviewing SEARCH PROGRESS (searching is still ongoing).
Decide ONLY whether the CONTEXT ALREADY COLLECTED lets every sub-step of the
person's planning be answered end-to-end. You are NOT judging the drafts'
quality — only whether the NEEDED FACTS ARE PRESENT.

Question:
{{question}}

Claims (each a distilled, evidence-backed report; "Evidence" is the cited
retrieved text):
{{claims_context}}

Overall rough draft assembled from the claims:
{{overall_draft}}

RECALL-ORIENTED GATE (read carefully — this is where auditors are wrong most
often):
1. Sufficient == the drafted answer already contains the specific facts the
   question asks for. Do NOT demand more depth, more comparisons, more
   caveats, or corroboration. "Present but thin" is SUFFICIENT.
2. Prefer NOT sufficient when the question asks for something concrete that is
   nowhere in the context (a name, date, number, mechanism, or relation).
3. Treat a solid single-source finding as sufficient when the question asks for
   a fact. Multiple sources are not required.
4. Judge the CONTEXT, not the drafts' phrasing: rough, partial, or unpolished
   drafts still count as sufficient when the underlying facts are present.
5. Judge against the OVERALL DRAFT too: per-claim sufficiency does not imply
   the whole question can be answered.

Output format (JSON only):
{
  "is_sufficient": true,
  "confidence": 0.85,
  "contradictions": [],
  "reasoning": "…",
  "sub_queries": [
    {"sub_query": "…", "satisfied": true, "missing_fact": "", "search_hint": ""}
  ],
  "claims": [
    {
      "claim_id": "…",
      "grounded": true,
      "ungrounded_assertions": [{"assertion": "…", "reason": "…"}],
      "missing_information": [{"what": "…", "search_hint": "…"}]
    }
  ]
}

When is_sufficient is false, the claims array MUST contain a missing_information
entry for every claim that is not yet grounded, and you SHOULD fill sub_queries
with the concrete steps that are still unsatisfied.`

// fallbackQueryRewrite is a CONDENSED restatement of sca_query_rewrite.md — the
// canonical copy is the embedded .md (full multi-hop / diversity rule set), and
// this constant exists only so a loader that cannot resolve the name still
// degrades to a working rewrite rather than a failure.
const fallbackQueryRewrite = `You are a Query Rewriter for a multi-hop RAG system.
Turn the Sufficient Context Agent's missing-pieces feedback into targeted,
retrievable search queries.

Original user question:
{{ question }}

Already-resolved bridge values (anchor the new query to these):
{{ bridge_values }}

Research history and evidence at hand:
{{ research_context }}

Missing pieces (each: "what" = what is still needed, "hint" = a search hint):
{{ gaps }}

Rules:
1. Name the specific missing entity + the relation/property needed.
2. MULTI-HOP: anchor the query to the already-resolved bridge value instead of
   re-deriving it.
3. Keep each query standalone — no pronouns; repeat the key entity explicitly.
4. ONE query per distinct missing piece; drop un-searchable pieces.
5. DIVERSITY: never paraphrase an already-tried query from the history.

Output JSON only, with no commentary:
{"queries": [{"query": "concrete search query 1"}, {"query": "..."}]}`
