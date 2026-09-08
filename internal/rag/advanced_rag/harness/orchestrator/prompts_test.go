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
	"strings"
	"testing"

	"ragflow/internal/rag/advanced_rag/harness"
)

// stubLoader serves a canned template, mirroring how tests wire
// harness.StringPromptLoader.
type stubLoader struct{ text string }

func (s stubLoader) Load(string) (string, error) { return s.text, nil }

// TestRenderPromptNilLoaderUsesEmbeddedCanonical locks in that an unset loader
// now resolves the canonical embedded .md (1:1 with rag/prompts/sca_select.md)
// rather than the condensed constant — the whole point of the orchestration
// parity fix.
func TestRenderPromptNilLoaderUsesEmbeddedCanonical(t *testing.T) {
	vars := map[string]string{"question": "Q?", "overall_draft": "D", "claims_context": "C"}
	got := renderPrompt(nil, "sca_select", fallbackSCAReview, vars)

	for _, frag := range []string{
		`Google-style "Sufficient Context Agent"`, // canonical sca_select.md only
		"Q?",
		"D",
		"C",
	} {
		if !strings.Contains(got, frag) {
			t.Errorf("canonical sca_select render missing %q", frag)
		}
	}
	if strings.Contains(got, "{{") {
		t.Errorf("render left an unresolved placeholder: %q", got)
	}
}

// TestRenderPromptNilLoaderQueryRewriteCanonical is the rewrite-side twin:
// nil loader must resolve the full embedded sca_query_rewrite.md, not the
// condensed constant.
func TestRenderPromptNilLoaderQueryRewriteCanonical(t *testing.T) {
	vars := map[string]string{
		"question":         "Q?",
		"gaps":             "G",
		"bridge_values":    "B",
		"research_context": "R",
	}
	got := renderPrompt(nil, "sca_query_rewrite", fallbackQueryRewrite, vars)

	// Rule 1 wording exists only in the canonical template.
	if !strings.Contains(got, "The query must name the specific missing entity") {
		t.Errorf("canonical sca_query_rewrite render not used; got:\n%s", got)
	}
	for _, frag := range []string{"Q?", "G", "B", "R"} {
		if !strings.Contains(got, frag) {
			t.Errorf("canonical sca_query_rewrite render missing %q", frag)
		}
	}
	if strings.Contains(got, "{{") {
		t.Errorf("render left an unresolved placeholder: %q", got)
	}
}

// TestRenderPromptUnknownNameFallsBack: a loader that cannot resolve the name
// still degrades to the condensed constant with vars substituted.
func TestRenderPromptUnknownNameFallsBack(t *testing.T) {
	got := renderPrompt(stubLoader{}, "no_such_template", fallbackQueryRewrite,
		map[string]string{"question": "Q?", "gaps": "G"})
	if !strings.Contains(got, "Query Rewriter") {
		t.Errorf("expected the fallback constant, got:\n%s", got)
	}
	if !strings.Contains(got, "Q?") || !strings.Contains(got, "G") {
		t.Errorf("fallback render missing var values: %q", got)
	}
}

// TestRenderPromptMissingVarInterpolatesEmpty mirrors Python's default Jinja
// Undefined: an unprovided variable renders as the empty string, not as a
// literal {{ var }}.
func TestRenderPromptMissingVarInterpolatesEmpty(t *testing.T) {
	loader := stubLoader{text: "A={{present}} B={{ absent }} C={{absent}}"}
	got := renderPrompt(loader, "t", "fallback", map[string]string{"present": "x"})
	if want := "A=x B= C="; got != want {
		t.Errorf("render = %q, want %q", got, want)
	}
}

// TestRenderPromptSubstitutesBothPlaceholderSpellings: Jinja tolerates the
// spaced and unspaced forms; the renderer must handle both.
func TestRenderPromptSubstitutesBothPlaceholderSpellings(t *testing.T) {
	loader := stubLoader{text: "{{v}}|{{ v }}"}
	got := renderPrompt(loader, "t", "fallback", map[string]string{"v": "x"})
	if want := "x|x"; got != want {
		t.Errorf("render = %q, want %q", got, want)
	}
}

var _ harness.PromptLoader = stubLoader{}
