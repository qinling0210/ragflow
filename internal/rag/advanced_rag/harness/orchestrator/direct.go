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

// Package orchestrator holds the top-level pipeline stages that sit above the
// action session: the low-mode direct search, the Sufficient Context Agent
// (SCA) review, and the gap→query rewriter.
//
// Mirrors Python rag/advanced_rag/harness/orchestrator/{direct,
// sufficient_context,query_rewriter}.py.
package orchestrator

import (
	"context"
	"fmt"

	"ragflow/internal/common"
	"ragflow/internal/rag/advanced_rag/harness"
)

var _LOG = common.StdLogger()

// KeywordExtractor produces the entity/qualifier-weighted retrieval query.
//
// Python calls tools._extract_keywords_weighted(question) when the method
// exists (`hasattr` guard). The Go equivalent is a nil-able interface: when the
// caller does not provide one, the weighted query is simply omitted — a missing
// extractor must not fail the request.
type KeywordExtractor interface {
	ExtractKeywordsWeighted(ctx context.Context, question string) (retrievalQuery string, keywords string, err error)
}

// DirectDeps are the dependencies of DirectSearch.
type DirectDeps struct {
	KB *harness.Kbinfos
	// Search performs one hybrid search.
	Search harness.SearchFn
	// Keywords is optional; nil skips weighted-keyword extraction.
	Keywords KeywordExtractor
}

// DirectResult mirrors Python direct_search's return dict
// ({"empty_result": bool, "kbinfos": ...}).
type DirectResult struct {
	// EmptyResult is true when the single pass retrieved nothing, which the
	// caller turns into an "I don't have enough information" answer.
	EmptyResult bool
	KB          *harness.Kbinfos
}

// DirectSearch mirrors Python orchestrator/direct.py::direct_search: ONE hybrid
// search merged into kbinfos. This is the whole of `low` mode.
//
// The weighted retrieval query is ALWAYS attached: a problem-level search over
// the bare question is exactly where the entity must dominate the ranking. A
// failure to extract it is logged and ignored rather than propagated.
func DirectSearch(ctx context.Context, deps DirectDeps, question, keywords string) DirectResult {
	if deps.KB == nil {
		return DirectResult{EmptyResult: true}
	}
	// Python wraps the whole body in @in_phase("direct").
	ctx, done := harness.Phase(ctx, harness.PhaseDirect)
	defer done()

	retrievalQuery := ""
	if deps.Keywords != nil {
		rq, _, err := deps.Keywords.ExtractKeywordsWeighted(ctx, question)
		if err != nil {
			_LOG.Printf("[Direct] entity-weighted keyword extraction failed: %v", err)
		} else {
			retrievalQuery = rq
		}
	}
	_LOG.Printf("[Direct search] Looking up the knowledge base for: %q (keywords: %s)", question, keywords)

	if deps.Search == nil {
		return DirectResult{EmptyResult: true, KB: deps.KB}
	}
	chunks, aggs := deps.Search(ctx, harness.SearchParams{
		Question:       question,
		Keywords:       keywords,
		RetrievalQuery: retrievalQuery,
		UseCompiled:    true,
	})
	deps.KB.Merge(chunks, aggs)

	if !deps.KB.HasChunks() {
		_LOG.Printf("[Direct search] Found no matching passages.")
		return DirectResult{EmptyResult: true, KB: deps.KB}
	}
	return DirectResult{KB: deps.KB}
}

// JSONModel generates one JSON object from a rendered prompt.
//
// Mirrors Python rag.prompts.generator.gen_json(prompt, "Output:\n", chat_mdl),
// which sends the prompt as the system turn, "Output:\n" as the user turn, and
// parses the first JSON value out of the reply.
type JSONModel interface {
	GenJSON(ctx context.Context, prompt string) (any, error)
}

// String returns a human-readable form used by the SCA log lines.
func fmtClaimCount(n int) string { return fmt.Sprintf("%d claim draft(s)", n) }
