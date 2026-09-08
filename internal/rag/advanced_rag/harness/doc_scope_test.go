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
	"errors"
	"testing"
)

// stubVerifier reports a fixed known set, or fails when err is set.
type stubVerifier struct {
	known map[string]bool
	err   error
}

func (s stubVerifier) KnownDocIDs(_ context.Context, _, _ []string) (map[string]bool, error) {
	return s.known, s.err
}

func TestResolveDocScopeKeepsKnownIDs(t *testing.T) {
	got := resolveDocScope(context.Background(), SearchDeps{
		DocIDVerifier: stubVerifier{known: map[string]bool{"a": true}},
	}, []string{"a", "b"}, []string{"kb"}, _LOG)

	if len(got) != 1 || got[0] != "a" {
		t.Fatalf("scope = %v, want [a]", got)
	}
}

func TestResolveDocScopeDropsScopeWhenAllUnknown(t *testing.T) {
	// Python retrieve:633-635 — an entirely bogus scope falls back to
	// unfiltered retrieval instead of returning nothing.
	got := resolveDocScope(context.Background(), SearchDeps{
		DocIDVerifier: stubVerifier{known: map[string]bool{"z": true}},
	}, []string{"a", "b"}, []string{"kb"}, _LOG)

	if got != nil {
		t.Fatalf("scope = %v, want nil (unfiltered)", got)
	}
}

func TestResolveDocScopePassthroughWithoutVerifier(t *testing.T) {
	scope := []string{"a", "b"}
	got := resolveDocScope(context.Background(), SearchDeps{}, scope, []string{"kb"}, _LOG)

	if len(got) != len(scope) {
		t.Fatalf("scope = %v, want %v unchanged", got, scope)
	}
}

func TestResolveDocScopePassthroughOnVerifierError(t *testing.T) {
	scope := []string{"a"}
	got := resolveDocScope(context.Background(), SearchDeps{
		DocIDVerifier: stubVerifier{err: errors.New("db down")},
	}, scope, []string{"kb"}, _LOG)

	if len(got) != 1 {
		t.Fatalf("scope = %v, want %v (verification unavailable)", got, scope)
	}
}

func TestDocInDatasetsRejectsForeignDocument(t *testing.T) {
	belongs, verified := docInDatasets(context.Background(), SearchDeps{
		DocIDVerifier: stubVerifier{known: map[string]bool{"other": true}},
	}, "doc")

	if !verified || belongs {
		t.Fatalf("belongs=%v verified=%v, want false/true (doc outside the datasets)", belongs, verified)
	}
}

func TestDocInDatasetsAcceptsOwnDocument(t *testing.T) {
	belongs, verified := docInDatasets(context.Background(), SearchDeps{
		DocIDVerifier: stubVerifier{known: map[string]bool{"doc": true}},
	}, "doc")

	if !verified || !belongs {
		t.Fatalf("belongs=%v verified=%v, want true/true", belongs, verified)
	}
}

func TestDocInDatasetsUnverifiedWithoutVerifier(t *testing.T) {
	// Without a verifier the caller must keep the previous behaviour instead of
	// rejecting the document.
	belongs, verified := docInDatasets(context.Background(), SearchDeps{}, "doc")

	if !belongs || verified {
		t.Fatalf("belongs=%v verified=%v, want true/false", belongs, verified)
	}
}

func TestDocInDatasetsUnverifiedOnError(t *testing.T) {
	belongs, verified := docInDatasets(context.Background(), SearchDeps{
		DocIDVerifier: stubVerifier{err: errors.New("db down")},
	}, "doc")

	if !belongs || verified {
		t.Fatalf("belongs=%v verified=%v, want true/false (lookup failed)", belongs, verified)
	}
}

func TestNavigateStructureDocRejectsForeignDocument(t *testing.T) {
	// Python _load_compiled_structure: a doc_id outside the bound datasets
	// yields no structure, so nothing from the other dataset is ever read.
	res, _ := navigateStructureDoc(context.Background(), "t", "q", "foreign-doc", "catalog", SearchDeps{
		KbIDs:         []string{"kb"},
		DocIDVerifier: stubVerifier{known: map[string]bool{"mine": true}},
	})

	if res.EmptyReason != ReasonNoStructure {
		t.Fatalf("EmptyReason = %q, want %q", res.EmptyReason, ReasonNoStructure)
	}
	if len(res.DocIDs) != 0 {
		t.Fatalf("DocIDs = %v, want none", res.DocIDs)
	}
}
