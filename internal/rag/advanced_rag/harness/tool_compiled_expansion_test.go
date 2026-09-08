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
	"strings"
	"testing"
)

// stubCompiledStore is an in-memory CompiledStore so the expander logic can be
// exercised in the unit tier. It behaves like the doc store:
//
//   - rows carry knowledge_graph_kwd / compile_kwd / available_int / from_entity_kwd
//     / to_entity_kwd / name_kwd / content_with_weight / source_chunk_ids / doc_id.
//   - filters are AND-ed, each value an OR-set (a row matches if its field's
//     string form equals a member).
//   - matchText selects rows whose name/content contains it (stand-in for the
//     dense+keyword retrieval the store normally does).
type stubCompiledStore struct {
	rows   []map[string]any
	chunks map[string]string
}

func (s *stubCompiledStore) SearchCompiled(_ context.Context, _, _ string, _ []string, filters map[string][]string, matchText string, topN int) ([]map[string]any, error) {
	var out []map[string]any
	for _, r := range s.rows {
		if !rowMatchesFilters(r, filters) {
			continue
		}
		if matchText != "" {
			hay := strings.ToLower(asString(r["name_kwd"]) + " " + asString(r["content_with_weight"]))
			if !strings.Contains(hay, strings.ToLower(matchText)) {
				continue
			}
		}
		out = append(out, r)
		if topN > 0 && len(out) >= topN {
			break
		}
	}
	return out, nil
}

func (s *stubCompiledStore) LoadChunks(_ context.Context, _, _ string, chunkIDs []string) ([]map[string]any, error) {
	var out []map[string]any
	for _, id := range chunkIDs {
		if content, ok := s.chunks[id]; ok {
			out = append(out, map[string]any{"id": id, "content": content, "doc_id": "d1"})
		}
	}
	return out, nil
}

func (s *stubCompiledStore) Vectorize(_ context.Context, _ string) ([]float64, error) {
	return nil, nil
}

func rowMatchesFilters(r map[string]any, filters map[string][]string) bool {
	for k, vals := range filters {
		rv := asString(r[k])
		matched := false
		for _, v := range vals {
			if rv == v {
				matched = true
				break
			}
		}
		if !matched {
			return false
		}
	}
	return true
}

func compiledIDs(chunks []map[string]any) []string {
	out := make([]string, 0, len(chunks))
	for _, c := range chunks {
		out = append(out, asString(c["id"]))
	}
	return out
}

// TestSeedNameFromRowParsesContentWithWeight mirrors Python L293-300: the seed
// name comes from content_with_weight JSON (name, else title), never from name_kwd.
func TestSeedNameFromRowParsesContentWithWeight(t *testing.T) {
	if got := seedNameFromRow(map[string]any{"content_with_weight": `{"name":"Culdcept","title":"Game"}`}); got != "Culdcept" {
		t.Errorf("name: got %q", got)
	}
	if got := seedNameFromRow(map[string]any{"content_with_weight": `{"title":"Released 1984"}`}); got != "Released 1984" {
		t.Errorf("title fallback: got %q", got)
	}
	if got := seedNameFromRow(map[string]any{"content_with_weight": `not json`}); got != "" {
		t.Errorf("bad json: got %q", got)
	}
	// name_kwd is not consulted — Python reads only the JSON payload.
	if got := seedNameFromRow(map[string]any{"content_with_weight": `{}`, "name_kwd": "X"}); got != "" {
		t.Errorf("name_kwd must be ignored, got %q", got)
	}
}

func TestLowerUnionSorted(t *testing.T) {
	got := lowerUnionSorted(map[string]bool{"OmiyaSoft": true, "X": true})
	// Byte-order ascending over the original + lowercased forms.
	want := []string{"OmiyaSoft", "X", "omiyasoft", "x"}
	if len(got) != len(want) {
		t.Fatalf("lowerUnionSorted = %v", got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("lowerUnionSorted[%d] = %q, want %q (%v)", i, got[i], want[i], got)
		}
	}
}

// TestExpandEntityStrategyOneHop verifies the 1-hop entity strategy: a seed
// entity matching the query (found via its template kind) walks a forward
// relation to a neighbour, resolves the neighbour by exact name_kwd, and loads
// its source chunks.
func TestExpandEntityStrategyOneHop(t *testing.T) {
	store := &stubCompiledStore{
		rows: []map[string]any{
			{"knowledge_graph_kwd": "entity", "compilation_template_kind_kwd": "knowledge_graph",
				"name_kwd": "Culdcept", "content_with_weight": `{"name":"Culdcept"}`},
			{"knowledge_graph_kwd": "relation", "compilation_template_kind_kwd": "knowledge_graph",
				"from_entity_kwd": "Culdcept", "to_entity_kwd": "OmiyaSoft"},
			{"knowledge_graph_kwd": "entity", "compilation_template_kind_kwd": "knowledge_graph",
				"name_kwd": "OmiyaSoft", "content_with_weight": `{"name":"OmiyaSoft"}`,
				"source_chunk_ids": []string{"s1", "s2"}},
		},
		chunks: map[string]string{"s1": "a", "s2": "b"},
	}
	exp := NewCompiledExpander(store, []string{"kb1"}, "t1").(*compiledExpander)
	kb := &Kbinfos{}
	if err := exp.Expand(context.Background(), kb, "culdcept", "", nil); err != nil {
		t.Fatalf("Expand: %v", err)
	}
	if len(kb.Chunks) != 2 {
		t.Fatalf("expanded = %d (%v), want 2 neighbour source chunks", len(kb.Chunks), compiledIDs(kb.Chunks))
	}
}

// TestExpandBackwardRelation confirms the strategy also walks incoming
// (to_entity_kwd) relations, mirroring Python's fwd+bwd merge.
func TestExpandBackwardRelation(t *testing.T) {
	store := &stubCompiledStore{
		rows: []map[string]any{
			{"knowledge_graph_kwd": "entity", "compilation_template_kind_kwd": "mind_map",
				"name_kwd": "Culdcept", "content_with_weight": `{"name":"Culdcept"}`},
			// Relation where the seed is the TARGET, not the source.
			{"knowledge_graph_kwd": "relation", "compilation_template_kind_kwd": "mind_map",
				"from_entity_kwd": "OmiyaSoft", "to_entity_kwd": "Culdcept"},
			{"knowledge_graph_kwd": "entity", "compilation_template_kind_kwd": "mind_map",
				"name_kwd": "OmiyaSoft", "content_with_weight": `{"name":"OmiyaSoft"}`,
				"source_chunk_ids": []string{"o1"}},
		},
		chunks: map[string]string{"o1": "from"},
	}
	exp := NewCompiledExpander(store, []string{"kb1"}, "t1").(*compiledExpander)
	kb := &Kbinfos{}
	if err := exp.Expand(context.Background(), kb, "culdcept", "", nil); err != nil {
		t.Fatalf("Expand: %v", err)
	}
	if len(kb.Chunks) != 1 {
		t.Fatalf("backward relation gave %d (%v), want 1", len(kb.Chunks), compiledIDs(kb.Chunks))
	}
}

// TestExpandWikiSetsSimilarityAndBlendsByIt verifies wiki synthesis pages load
// their source chunks with a similarity boost (0.9) and the final sort puts the
// higher-similarity wiki chunk ahead of the entity (0-similarity) chunk.
func TestExpandWikiSetsSimilarityAndBlendsByIt(t *testing.T) {
	store := &stubCompiledStore{
		rows: []map[string]any{
			// knowledge_graph entity 1-hop -> chunk e1 (similarity 0 by default).
			{"knowledge_graph_kwd": "entity", "compilation_template_kind_kwd": "knowledge_graph",
				"name_kwd": "Culdcept", "content_with_weight": `{"name":"Culdcept"}`},
			{"knowledge_graph_kwd": "relation", "compilation_template_kind_kwd": "knowledge_graph",
				"from_entity_kwd": "Culdcept", "to_entity_kwd": "OmiyaSoft"},
			{"knowledge_graph_kwd": "entity", "compilation_template_kind_kwd": "knowledge_graph",
				"name_kwd": "OmiyaSoft", "content_with_weight": `{"name":"OmiyaSoft"}`,
				"source_chunk_ids": []string{"e1"}},
			// wiki synthesis page matching the query -> chunk w1.
			{"compile_kwd": "wiki_page", "available_int": "1", "doc_id": "d2",
				"name_kwd": "Culdcept", "content_with_weight": `{"title":"Culdcept article"}`,
				"source_chunk_ids": []string{"w1"}},
		},
		chunks: map[string]string{"e1": "entity", "w1": "wiki"},
	}
	exp := NewCompiledExpander(store, []string{"kb1"}, "t1").(*compiledExpander)
	kb := &Kbinfos{}
	if err := exp.Expand(context.Background(), kb, "culdcept", "", nil); err != nil {
		t.Fatalf("Expand: %v", err)
	}
	if len(kb.Chunks) != 2 {
		t.Fatalf("expanded = %d (%v), want 2", len(kb.Chunks), compiledIDs(kb.Chunks))
	}
	var wSim, eSim float64
	for _, c := range kb.Chunks {
		switch asString(c["id"]) {
		case "w1":
			wSim = similarityOf(c)
		case "e1":
			eSim = similarityOf(c)
		}
	}
	if wSim != 0.9 {
		t.Errorf("wiki similarity = %v, want 0.9", wSim)
	}
	if eSim != 0.0 {
		t.Errorf("entity chunk similarity = %v, want 0.0", eSim)
	}
	// Final global sort: wiki (0.9) must rank before entity (0).
	if compiledIDs(kb.Chunks)[0] != "w1" {
		t.Errorf("final sort should put wiki (0.9) first, got %v", compiledIDs(kb.Chunks))
	}
}

// TestExpandCapsMaxChunksAndSkipsSeen confirms a strategy stops adding at 5 and
// chunks already in kbinfos are not duplicated.
func TestExpandCapsMaxChunksAndSkipsSeen(t *testing.T) {
	store := &stubCompiledStore{
		rows: []map[string]any{
			{"knowledge_graph_kwd": "entity", "compilation_template_kind_kwd": "knowledge_graph",
				"name_kwd": "A", "content_with_weight": `{"name":"A"}`},
			{"knowledge_graph_kwd": "relation", "compilation_template_kind_kwd": "knowledge_graph",
				"from_entity_kwd": "A", "to_entity_kwd": "B"},
			{"knowledge_graph_kwd": "entity", "compilation_template_kind_kwd": "knowledge_graph",
				"name_kwd": "B", "content_with_weight": `{"name":"B"}`,
				"source_chunk_ids": []string{"b1", "b2", "b3", "b4", "b5", "b6", "b7"}},
		},
		chunks: map[string]string{"b1": "1", "b2": "2", "b3": "3", "b4": "4", "b5": "5", "b6": "6", "b7": "7"},
	}
	exp := NewCompiledExpander(store, []string{"kb1"}, "t1").(*compiledExpander)
	// b1 is already present in the working set; it must not come back.
	kb := &Kbinfos{Chunks: []map[string]any{{"id": "b1", "content": "pre-existing"}}}
	if err := exp.Expand(context.Background(), kb, "a", "", nil); err != nil {
		t.Fatalf("Expand: %v", err)
	}
	// 5 fresh (max_chunks) minus the seen b1 that was skipped within the first
	// load batch -> at most 5 new.
	if len(kb.Chunks) > 6 { // 1 pre-existing + up to 5 new
		t.Fatalf("exceeded max_chunks=5 new chunks: got %d total (%v)", len(kb.Chunks), compiledIDs(kb.Chunks))
	}
	for _, c := range kb.Chunks {
		if asString(c["id"]) == "b1" && asString(c["content"]) != "pre-existing" {
			t.Error("seen chunk b1 was duplicated")
		}
	}
}
