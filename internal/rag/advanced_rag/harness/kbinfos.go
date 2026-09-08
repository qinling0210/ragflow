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
	"fmt"
	"sync"
)

// The shared accumulation store and the retrieval abstraction.
//
// In Python this is not a harness module: it is the plain dict
// `self.kbinfos = {"chunks": [], "doc_aggs": [], "memory": []}` carried by the
// RAGTools instance in agentic_rag.py:307, threaded through retrieval, the
// action session, SCA, and the final answer via tools.kbinfos. The Go port
// reifies that dict into the typed Kbinfos struct (with the Merge helper) and
// keeps it in the harness package because chunkKey/chunkText and Merge are
// unexported helpers shared by tool_search.go, tool_exploration.go, and
// tool_compiled_expansion.go — a reification, not a 1:1 file mirror.
//
// This file also owns the chunk identity helpers (chunkKey/chunkText) that the
// merge logic depends on.

// SearchParams mirrors Python tools/search.py::hybrid_search(tools, query,
// kb_ids, top_n, doc_scope, keywords, retrieval_query, use_compiled).
//
// Question is the search query; Keywords is used ONLY to narrow retrieved
// chunks (never to build the query) unless RetrievalQuery is empty.
type SearchParams struct {
	Question string
	// Keywords narrows retrieved chunks to the sentences that mention them.
	// When RetrievalQuery is empty it also falls back into the query text.
	Keywords string
	// RetrievalQuery is the entity/qualifier-weighted query (entity x3,
	// qualifier x3) appended to the bare Question: a problem-level search over
	// the raw question is exactly where the entity must dominate the ranking.
	RetrievalQuery string
	// UseCompiled enables compiled-structure expansion (page index / tree /
	// mindmap / knowledge graph / wiki synthesis pages).
	UseCompiled bool
	// KbIDs restricts the search to these dataset ids. Empty falls back to the
	// session's bound datasets.
	KbIDs []string
	// DocScope restricts the search to these document ids (e.g. the doc_id
	// routed by navigate_tree).
	DocScope []string
	// TopN is the result count. <= 0 selects the configured default.
	TopN int
	// HybridTool marks this call as the standalone hybrid_search tool
	// (Python search_chunks → hybrid_search), as opposed to RAGTools.retrieve.
	// It excludes compiled-product rows from the base retrieval (must_not
	// compile_kwd) and defaults the vector weight to 0.3 (Python
	// _DEFAULT_HYBRID_VECTOR_WEIGHT), whereas retrieve neither excludes them
	// nor defaults below 0.7.
	HybridTool bool
}

// SearchFn performs one hybrid search and returns chunks + doc aggs, so the
// harness is decoupled from the concrete retrieval backend.
// Mirrors the Python side's settings.retriever.retrieval call.
type SearchFn func(ctx context.Context, p SearchParams) ([]map[string]any, []map[string]any)

// Kbinfos is the shared accumulation store.
type Kbinfos struct {
	Chunks  []map[string]any
	DocAggs []map[string]any
	// Memory is the lossless store of raw retrieved chunks backing the (lossy)
	// Chunks list that feeds the LLM. Mirrors Python kbinfos["memory"], which
	// memory.add/grep maintain.
	Memory []map[string]any
	// PreSummary is the merged claim-report summary produced by the action
	// session; the final-answer call reads it when set.
	PreSummary string
	// cache is the per-request retrieval cache (Python tools.search_cache). It
	// is initialised lazily via cacheOnce so a zero-value Kbinfos is usable.
	cache     *searchCache
	cacheOnce sync.Once
}

// HasChunks mirrors Python `bool(kbinfos.get("chunks"))`.
func (k *Kbinfos) HasChunks() bool { return len(k.Chunks) > 0 }

// Merge appends the given chunks/aggs, deduplicating by chunkKey. It returns
// the GLOBAL indices (positions in k.Chunks after the merge) of the chunks this
// call contributed, so callers can store stable evidence references across
// multiple Merge calls — per-search indices would diverge from the accumulated
// list.
//
// NOTE: one entry is returned per CONTRIBUTED chunk, so a chunk that was
// already present still yields its index. len(result) is therefore NOT a count
// of new evidence — callers that need "how many were actually added" must
// compare len(k.Chunks) before and after.
func (k *Kbinfos) Merge(chunks, aggs []map[string]any) []int {
	seen := make(map[string]bool, len(k.Chunks))
	for _, c := range k.Chunks {
		seen[chunkKey(c)] = true
	}
	var added []int
	for _, c := range chunks {
		kk := chunkKey(c)
		if !seen[kk] {
			seen[kk] = true
			k.Chunks = append(k.Chunks, c)
		}
		// Record the global index of every contributed chunk (dedup or new) so
		// EvidenceIDs always reference accumulated kbinfos positions.
		added = append(added, indexOfChunk(k.Chunks, kk))
	}
	dseen := make(map[string]bool, len(k.DocAggs))
	for _, d := range k.DocAggs {
		if id, _ := d["doc_id"].(string); id != "" {
			dseen[id] = true
		}
	}
	for _, d := range aggs {
		if id, _ := d["doc_id"].(string); id != "" && !dseen[id] {
			dseen[id] = true
			k.DocAggs = append(k.DocAggs, d)
		}
	}
	return added
}

// indexOfChunk returns the global index of the chunk whose key matches kk.
func indexOfChunk(chunks []map[string]any, kk string) int {
	for i, c := range chunks {
		if chunkKey(c) == kk {
			return i
		}
	}
	return -1
}

// chunkText mirrors Python _chunk_text: the searchable text of a chunk,
// preferring the weighted (reranked) content, then raw content, then "text".
// chunkText returns the searchable text of a chunk. Mirrors Python
// memory._chunk_text: prefer "content" over "content_with_weight" (Python
// checks them in that order). A "text" fallback is kept as a Go-only superset
// for chunks that carry neither key.
func chunkText(c map[string]any) string {
	if t, ok := c["content"].(string); ok && t != "" {
		return t
	}
	if t, ok := c["content_with_weight"].(string); ok && t != "" {
		return t
	}
	if t, ok := c["text"].(string); ok {
		return t
	}
	return ""
}

// chunkKey returns a stable dedup key for a chunk. Prefers chunk_id/id when
// present; otherwise falls back to a content-derived key so chunks without an
// id still dedup correctly (and never collide on a shared empty key).
//
// This diverges from Python _chunk_key (`chunk_id or id or id(ck)`): Python's
// final fallback is `id(ck)` — the dict object's memory address — which means
// two equivalent chunks returned as distinct objects never share a key, so
// deduplication silently fails for id-less chunks. Go instead keys on the chunk
// text, so identical content still merges across retrieval calls.
func chunkKey(c map[string]any) string {
	if id, ok := c["chunk_id"].(string); ok && id != "" {
		return "cid:" + id
	}
	if id, ok := c["id"].(string); ok && id != "" {
		return "id:" + id
	}
	content := ""
	if t, ok := c["content_with_weight"].(string); ok {
		content = t
	} else if t, ok := c["content"].(string); ok {
		content = t
	}
	if content == "" {
		// No stable identity: fall back to the doc reference so at least
		// per-document grouping is preserved (rarely reached).
		content = fmt.Sprintf("%s|%s", anyString(c["doc_id"]), anyString(c["docnm_kwd"]))
	}
	return "h:" + fnv64(content)
}

func anyString(v any) string {
	if s, ok := v.(string); ok {
		return s
	}
	return ""
}

// fnv64 is a deterministic non-crypto hash for dedup keys.
func fnv64(s string) string {
	h := uint64(14695981039346656037)
	for i := 0; i < len(s); i++ {
		h ^= uint64(s[i])
		h *= 1099511628211
	}
	return fmt.Sprintf("%016x", h)
}
