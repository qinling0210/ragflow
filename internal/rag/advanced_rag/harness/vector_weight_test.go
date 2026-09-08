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

import "testing"

func TestAgenticVectorWeightDefaultsToZero(t *testing.T) {
	// Python's agentic retrieve runs keyword-only: using_embedding defaults to
	// False and no caller passes True (agentic_rag.py:600, 643-646).
	got := floatPtrOrDef(nil, DefaultAgenticVectorWeight)
	if got != 0 {
		t.Fatalf("default vector weight = %v, want 0 (keyword-only, as in Python)", got)
	}
	if DefaultAgenticVectorWeight != 0 {
		t.Fatalf("DefaultAgenticVectorWeight = %v, want 0", DefaultAgenticVectorWeight)
	}
}

func TestVectorWeightHonoursExplicitZero(t *testing.T) {
	// A pointer keeps "configured 0" distinct from "unset", so keyword-only can
	// be requested explicitly rather than only by omission.
	zero := 0.0
	if got := floatPtrOrDef(&zero, DefaultAgenticVectorWeight); got != 0 {
		t.Fatalf("got %v, want 0", got)
	}

	// Fallback must apply only when unset.
	if DefaultAgenticVectorWeight != 0 {
		t.Fatal("the fallback itself must stay 0 to match Python")
	}
}

func TestVectorWeightCanEnableHybrid(t *testing.T) {
	// Hybrid retrieval stays available, just not by default.
	hybrid := 0.3
	if got := floatPtrOrDef(&hybrid, DefaultAgenticVectorWeight); got != 0.3 {
		t.Fatalf("got %v, want 0.3 when explicitly configured", got)
	}
}

func TestRetrievalDefaultsUseIntOrDef(t *testing.T) {
	if got := intOrDef(0, 12); got != 12 {
		t.Fatalf("intOrDef(0) = %d, want the fallback 12", got)
	}
	if got := intOrDef(5, 12); got != 5 {
		t.Fatalf("intOrDef(5) = %d, want 5", got)
	}
}

func TestResolveVectorWeightMirrorsUsingEmbedding(t *testing.T) {
	// Python RAGTools.retrieve(using_embedding: bool = False) (agentic_rag.py:599).
	// Off → keyword-only (weight 0); on → 0.7 default or the configured override.
	if got := resolveVectorWeight(SearchDeps{UsingEmbedding: false}, false); got != 0 {
		t.Fatalf("using_embedding=false → %v, want 0 (keyword-only)", got)
	}
	if got := resolveVectorWeight(SearchDeps{UsingEmbedding: true}, false); got != DefaultHybridVectorWeight {
		t.Fatalf("using_embedding=true → %v, want %v", got, DefaultHybridVectorWeight)
	}
	override := 0.5
	if got := resolveVectorWeight(SearchDeps{UsingEmbedding: true, VectorSimilarityWeight: &override}, false); got != 0.5 {
		t.Fatalf("using_embedding=true with override → %v, want 0.5", got)
	}
}

func TestResolveVectorWeightHybridToolDefaultsToThreeTenths(t *testing.T) {
	// Python hybrid_search defaults the vector weight to 0.3
	// (_DEFAULT_HYBRID_VECTOR_WEIGHT, search.py:51), unlike RAGTools.retrieve's 0.7.
	if got := resolveVectorWeight(SearchDeps{UsingEmbedding: true}, true); got != HybridSearchDefaultVectorWeight {
		t.Fatalf("hybrid tool → %v, want %v", got, HybridSearchDefaultVectorWeight)
	}
	if got := resolveVectorWeight(SearchDeps{UsingEmbedding: false}, true); got != 0 {
		t.Fatalf("hybrid tool with embedding off → %v, want 0", got)
	}
}
