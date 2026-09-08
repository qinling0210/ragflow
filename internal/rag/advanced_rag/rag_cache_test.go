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

package advanced_rag

import "testing"

func TestQuestionKeywordsSeparatesNumbers(t *testing.T) {
	gram := questionKeywords("Population of Paris in 2019")

	if !gram.numbers["2019"] {
		t.Fatalf("numbers = %v, want 2019 separated out", gram.numbers)
	}
	if gram.words["2019"] {
		t.Fatal("2019 must not count as a significant word")
	}
	if !gram.words["population"] || !gram.words["paris"] {
		t.Fatalf("words = %v, want population and paris", gram.words)
	}
	// Stopwords are dropped.
	if gram.words["of"] || gram.words["in"] {
		t.Fatalf("words = %v, stopwords must be dropped", gram.words)
	}
}

func TestCacheSimilarCollapsesReask(t *testing.T) {
	// Python's observed re-ask: "legal population" → "estimated population of
	// Paris in 2019" (overlap 0.75 while numbers match).
	a := questionKeywords("population of Paris 2019")
	b := questionKeywords("legal population of Paris in 2019")

	if !cacheSimilar(a, b) {
		t.Fatal("near-identical re-ask must be judged similar")
	}
}

func TestCacheSimilarRejectsDifferentNumbers(t *testing.T) {
	// Different years are different questions and must not share an answer.
	a := questionKeywords("population of Paris 2019")
	b := questionKeywords("population of Paris 2015")

	if cacheSimilar(a, b) {
		t.Fatal("questions naming different numbers must not be similar")
	}
}

func TestCacheSimilarRejectsDifferentSubjects(t *testing.T) {
	// Paris vs. Brown County ≈ 0.25 overlap.
	a := questionKeywords("population of Paris 2019")
	b := questionKeywords("population of Brown County 2019")

	if cacheSimilar(a, b) {
		t.Fatal("genuinely different questions must not be similar")
	}
}

func TestCacheLookupAndStore(t *testing.T) {
	c := NewRAGCache()
	c.Store("population of Paris 2019", "2.1 million")

	got, ok := c.Lookup("legal population of Paris in 2019")
	if !ok || got != "2.1 million" {
		t.Fatalf("Lookup = %q, %v; want the cached answer", got, ok)
	}
}

func TestCacheLookupMissesUnrelatedQuestion(t *testing.T) {
	c := NewRAGCache()
	c.Store("population of Paris 2019", "2.1 million")

	if _, ok := c.Lookup("who wrote the book about Brown County"); ok {
		t.Fatal("unrelated question must not hit the cache")
	}
}

func TestCacheReuseBlockedAfterInsufficientRound(t *testing.T) {
	// Python :848-852 — when the last round was not SUFFICIENT the caller is
	// asking again for more evidence, so the cached answer must not be reused.
	c := NewRAGCache()
	c.Store("population of Paris 2019", "2.1 million")
	c.noteVerdict("INSUFFICIENT")

	if _, ok := c.Lookup("legal population of Paris in 2019"); ok {
		t.Fatal("reuse must be blocked after an INSUFFICIENT round")
	}

	c.noteVerdict("SUFFICIENT")
	if _, ok := c.Lookup("legal population of Paris in 2019"); !ok {
		t.Fatal("reuse must resume once the round is sufficient")
	}
}

func TestNilCacheIsInert(t *testing.T) {
	var c *RAGCache

	if _, ok := c.Lookup("anything"); ok {
		t.Fatal("nil cache must not hit")
	}
	c.Store("q", "a") // must not panic
	c.noteVerdict("SUFFICIENT")
}

func TestResolveEffectiveQuestionPrefersOriginal(t *testing.T) {
	// The outer rewrite dropped the final target of a multi-hop question.
	got := resolveEffectiveQuestion(
		"when did the purchaser die",
		"when did the purchaser of the shortest abbreviation die",
	)

	if got != "when did the purchaser of the shortest abbreviation die" {
		t.Fatalf("got %q, want the original question", got)
	}
}

func TestResolveEffectiveQuestionKeepsDifferentTurn(t *testing.T) {
	// A genuine re-ask for a different question must keep its own query.
	got := resolveEffectiveQuestion("population of Brown County", "population of Paris 2019")

	if got != "population of Brown County" {
		t.Fatalf("got %q, want the rewrite kept for a different question", got)
	}
}

func TestResolveEffectiveQuestionHandlesEmpty(t *testing.T) {
	if got := resolveEffectiveQuestion("q", ""); got != "q" {
		t.Fatalf("got %q, want q", got)
	}
	if got := resolveEffectiveQuestion("", "orig"); got != "" {
		t.Fatalf("got %q, want empty", got)
	}
}
