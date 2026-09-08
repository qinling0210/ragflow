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

package prompts

import (
	"strings"
	"testing"
)

// TestEmbeddedLoaderCanonicalTemplates makes sure every template the Go column
// relies on is actually embedded and stripped the way Python's load_prompt
// strips the canonical rag/prompts/*.md files.
func TestEmbeddedLoaderCanonicalTemplates(t *testing.T) {
	names := []string{
		"action_run",
		"action_initialize_state",
		"sca_select",
		"sca_query_rewrite",
	}
	for _, name := range names {
		got, err := (EmbeddedPromptLoader{}).Load(name)
		if err != nil {
			t.Fatalf("Load(%q): %v", name, err)
		}
		if got == "" {
			t.Errorf("Load(%q) returned an empty template", name)
		}
		if strings.TrimSpace(got) != got {
			t.Errorf("Load(%q) returned surrounding whitespace (load_prompt strips both ends)", name)
		}
	}
}

// TestEmbeddedLoaderMissingName reports an error for templates we do not embed,
// so the caller's fallback engages exactly like load_prompt's FileNotFoundError.
func TestEmbeddedLoaderMissingName(t *testing.T) {
	if _, err := (EmbeddedPromptLoader{}).Load("no_such_template"); err == nil {
		t.Fatal("Load(no_such_template) succeeded, want error")
	}
}
