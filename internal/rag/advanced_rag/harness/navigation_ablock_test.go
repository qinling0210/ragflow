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
	"strings"
	"testing"
)

func TestVecsEqual(t *testing.T) {
	if !vecsEqual([]float64{1, 2, 3}, []float64{1, 2, 3}) {
		t.Error("identical vectors should be equal")
	}
	if vecsEqual([]float64{1, 2, 3}, []float64{1, 2, 4}) {
		t.Error("differing vectors should not be equal")
	}
	if vecsEqual([]float64{1, 2}, []float64{1, 2, 3}) {
		t.Error("differently-sized vectors should not be equal")
	}
	if vecsEqual(nil, []float64{1}) {
		t.Error("empty vs non-empty should not be equal")
	}
}

func TestHasDistinctNodeVectors(t *testing.T) {
	if hasDistinctNodeVectors([]structureNode{{name: "a"}, {name: "b"}}) {
		t.Error("no vectors at all -> not distinct")
	}
	if hasDistinctNodeVectors([]structureNode{{name: "a", vec: []float64{1, 2}}}) {
		t.Error("a single vector-bearing node -> not distinct")
	}
	// RAPTOR: every node inherits the same blob vector.
	if hasDistinctNodeVectors([]structureNode{
		{name: "a", vec: []float64{1, 2}},
		{name: "b", vec: []float64{1, 2}},
	}) {
		t.Error("all nodes sharing one vector -> not distinct")
	}
	// page_index: per-node vectors differ.
	if !hasDistinctNodeVectors([]structureNode{
		{name: "a", vec: []float64{1, 2}},
		{name: "b", vec: []float64{3, 4}},
	}) {
		t.Error("distinct per-node vectors -> distinct")
	}
	// One vector-bearing among several is treated as not-distinct (Python: <2 distinct).
	if hasDistinctNodeVectors([]structureNode{
		{name: "a", vec: []float64{1, 2}},
		{name: "b"},
	}) {
		t.Error("one vector-bearing among two -> not distinct")
	}
}

func TestNodesCoveringChunks(t *testing.T) {
	nodes := []structureNode{
		{name: "A", sourceChunkIDs: []string{"c1", "c2"}},
		{name: "B", sourceChunkIDs: []string{"c3"}},
		{name: "C"},
	}
	got := nodesCoveringChunks(nodes, []string{"c2", "c9"})
	if len(got) != 1 || got[0] != "A" {
		t.Errorf("covering = %v, want [A]", got)
	}
	if got := nodesCoveringChunks(nodes, nil); got != nil {
		t.Errorf("nil chunk list -> nil, got %v", got)
	}
}

func renderDrillNodes() ([]structureNode, []structureRel) {
	nodes := []structureNode{
		{name: "Root", nodeType: "tree_node", desc: "root entity", sourceChunkIDs: []string{"r1"}},
		{name: "Leaf", nodeType: "tree_node", desc: "leaf entity", sourceChunkIDs: []string{"l1"}},
	}
	rels := []structureRel{{from: "Root", to: "Leaf", relType: "contains"}}
	return nodes, rels
}

func drillLoader(ids []string) []chunkWithText {
	var out []chunkWithText
	for _, id := range ids {
		out = append(out, chunkWithText{id: id, text: "chunk content for " + id})
	}
	return out
}

func TestRenderTocDrilloutLLMSelect(t *testing.T) {
	nodes, rels := renderDrillNodes()
	out := renderTocDrilldown("some query", nil, nodes, rels, drillLoader, nil, []string{"Leaf"})
	if out.selector != "llm_toc" {
		t.Errorf("selector = %q, want llm_toc", out.selector)
	}
	if out.outline == "" {
		t.Fatal("outline should be non-empty for a selected leaf")
	}
	// The selected leaf's parent (Root) must be pulled in as an ancestor.
	if !strings.Contains(out.outline, "Root") {
		t.Errorf("ancestor Root missing from outline:\n%s", out.outline)
	}
}

func TestRenderTocDrilloutChunkRetrieval(t *testing.T) {
	nodes, rels := renderDrillNodes()
	hits := []chunkHit{{id: "r1", score: 0.9}, {id: "l1", score: 0.7}}
	out := renderTocDrilldown("q", nil, nodes, rels, drillLoader, hits, nil)
	if out.selector != "chunk_retrieval" {
		t.Errorf("selector = %q, want chunk_retrieval", out.selector)
	}
	if out.topScore != 0.9 {
		t.Errorf("topScore = %v, want 0.9", out.topScore)
	}
	// Retrieved chunks are surfaced even when they are not a selected node's anchor.
	if !strings.Contains(out.outline, "r1") || !strings.Contains(out.outline, "l1") {
		t.Errorf("retrieved chunks missing from outline:\n%s", out.outline)
	}
}

func TestRenderTocDrilloutBeamFallback(t *testing.T) {
	nodes, rels := renderDrillNodes()
	// No selection and no hits, but a query: beam picks by keyword relevance. The
	// blank query yields no terms, so it must still return a (flat) outline, not panic.
	out := renderTocDrilldown("leaf", nil, nodes, rels, drillLoader, nil, nil)
	if out.outline == "" {
		t.Fatal("beam path produced an empty outline")
	}
	if out.selector != "beam" {
		t.Errorf("selector = %q, want beam", out.selector)
	}
}

func TestParseCompiledStructureCarriesVec(t *testing.T) {
	rows := []StructureRow{{
		CompileKwd:        "tree",
		KnowledgeGraphKwd: "graph",
		Content:           `{"entities": [{"name": "A"}, {"name": "B"}], "relations": []}`,
		Vec:               []float64{1, 2},
	}}
	ents, _ := ParseCompiledStructure(rows, []string{"tree"})
	if len(ents) != 2 {
		t.Fatalf("entities = %d, want 2", len(ents))
	}
	// Both nested entities of the RAPTOR blob inherit its single vector.
	for _, e := range ents {
		v, _ := e["_vec"].([]float64)
		if len(v) != 2 || v[0] != 1 {
			t.Errorf("entity %v did not inherit the blob vector", e["name"])
		}
	}
}
