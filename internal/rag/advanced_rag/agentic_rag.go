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

// Package advanced_rag is the outer agentic-search loop (medium / high / ultra).
//
// This file mirrors Python rag/advanced_rag/agentic_rag.py — the RAGTools
// methods: the run configuration (RAGTools), the rag entry point (Rag)
// and question formalization (Formalize). In Python, RAGTools is the shared
// capability carrier that the agentic graph (agentic_rag_graph.py) and the
// harness action session both read from and write to. In Go the equivalent
// carrier is harness.Toolset (see harness/action_session.go), while this file
// holds the RAGTools-side logic.
//
// The graph driver itself lives in agentic_rag_graph.go (mirroring
// agentic_rag_graph.py), so this package replicates the Python layout:
//   - agentic_rag.go      ↔ Python agentic_rag.py            (RAGTools methods)
//   - agentic_rag_graph.go ↔ Python agentic_rag_graph.py      (the pipeline)
//   - harness/            ↔ Python harness/                   (primitives)
package advanced_rag

import (
	"context"
	"fmt"
	"log"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/cloudwego/eino/schema"

	"ragflow/internal/agent/chat"
	"ragflow/internal/dao"
	"ragflow/internal/entity"
	"ragflow/internal/rag/advanced_rag/harness"
	"ragflow/internal/rag/prompts"
	"ragflow/internal/service/nlp"
)

// RAGTools method map — where each Python method lives in Go.
//
// Python's RAGTools is one object exposing every capability. Go keeps the same
// shape: the RAGTools methods live in this package (agentic_rag_graph.go and
// this file), while harness/ stays the
// leaf capability library they orchestrate (retrieval, sessions, tools). Nothing
// below is re-implemented as a forwarding shim — the Go column IS the
// implementation:
//
//	__init__                221 → RAGTools (below)
//	has_*                   340 → harness.Toolset            (action_session.go)
//	_fit_messages           360 → chat.FitMessages           (chat/message_fit.go)
//	get_citation_guidelines 367 → GetCitationGuidelines      (below)
//	sys_prompt              371 → SysPrompt                  (below)
//	formalize               412 → Formalize                  (below)
//	retrieve                593 → HybridSearch               (harness/tools/search.go)
//	_extract_keywords_weighted 571 → ExtractWeightedKeywords (harness/keywords.go)
//	extract_keywords        581 → ExtractWeightedKeywords (harness/keywords.go) + CompactKeywords (harness/tool_text_processing.go)
//	web_retrieve            676 → searchExecutor.webSearch   (harness/tool_executor.go)
//	_compose_answer_from_evidence 604 → ComposeAnswer        (agentic_rag_graph.go)
//	_naive_rag                   1517 → ComposeNaiveAnswer   (agentic_rag_graph.go)
//	_fit_evidence           716 → FitEvidence                (below)
//	judge_sufficiency       734 → SCA review                 (harness/orchestrator/sufficient_context.go)
//	gen_followups           750 → SCA gap rewrite            (harness/orchestrator/sufficient_context.go)
//	rag                     817 → Rag                       (below)
//
// Also mirrored, outside this file:
//
//	scoped_doc_ids           352 → toolDocScope          (harness/tool_executor.go)
//	_filter_known_doc_ids    981 → DocIDLookup           (below)
//	_resolve_doc_tenant      987 → docInDatasets         (harness/search.go)
//	fetch_full_document      761 → fetchFullDocument     (harness/doc_fetch.go)
//	summarize_document       934 → summarizeDocument     (harness/doc_fetch.go)
//
// All live Python methods now have a Go counterpart, including
// harness.LLMUsageStats (Python harness/stats.py::LLMUsageStats) — Python counts
// usage through a ContextVar and a wrapper chat model, so Go instead passes an
// explicit *harness.LLMUsageStats collector (RAGTools.Stats) to the calls it
// makes; the per-round metrics Python derives from RAGTools.rounds are not
// tracked.
//
// Deliberately not ported — dead code in Python: pick_documents returns None on
// its first statement (agentic_rag.py:500) and is never called, so
// _select_by_titles, _filter_by_metadata, _get_cached_metas and
// _collect_doc_titles are unreachable too; structured_retrieve has neither a
// @tool decorator nor a caller.
//
// RAGTools configures one agentic-search run. It is the single Go mirror of
// Python RAGTools.__init__: every capability the run needs (retriever, model,
// prompts, embedder, kb accumulation, retrieval tuning, answer composition) is
// carried on this one object — there is no second dependency struct. The
// harness layer keeps narrow per-call views (harness.SearchDeps /
// harness.SessionDeps) that are projected from a RAGTools at call time, mirroring
// how Python methods receive self plus extra per-call arguments.
type RAGTools struct {
	// Tools is the shared capability carrier (Python RAGTools instance). The
	// graph and the action session both reach retrieval through it.
	Tools *harness.Toolset
	// Retriever is the retrieval backend. When nil, the runtime singleton
	// (runtime.GetRetrievalService) is used.
	Retriever harness.Retriever
	// Model drives the LLM turns. Required for agentic modes; low/naive never
	// call it.
	Model harness.SessionModel
	// Embedder is the external embedding handle used by graph/structure seed
	// encoding (Python tools.embed_mdl). Nil falls back to this package's
	// internal tenant-default resolver, which additionally degrades to keyword
	// matching when the DB is uninitialised or encoding panics.
	Embedder nlp.NavEmbedder
	// Keywords extracts the entity-weighted retrieval query. Optional.
	Keywords KeywordExtractorFn
	// Prompts are the report/SCA/rewrite prompt templates (user_defined_prompts).
	Prompts harness.PromptLoader
	// Expand runs compiled-structure expansion. Optional.
	Expand harness.CompiledExpander
	// SCAPrompts overrides Prompts for the sufficient-context review.
	SCAPrompts harness.PromptLoader
	// RewritePrompts overrides Prompts for the gap→query rewrite call.
	RewritePrompts harness.PromptLoader
	// KB is the in-flight retrieval accumulation (Python RAGTools.kbinfos).
	KB *harness.Kbinfos
	// Logger is optional; nil uses the default logger.
	Logger *log.Logger
	// Answer-composition configuration (see AnswerDeps). All optional; the
	// defaults match Python's configured behaviour.
	// CiteRules overrides the default citation rules.
	CiteRules string
	// SystemPrompt is the dialog-level UI configuration appended after the
	// agentic contract (language/tone/style may override; evidence contract may not).
	SystemPrompt string
	// EmptyResponse is returned verbatim when no evidence was found, skipping
	// the composition call entirely (Python tools.empty_response).
	EmptyResponse string
	// EvidenceMaxTokens caps the evidence block. <=0 uses evidenceBudgetTokens.
	EvidenceMaxTokens int
	// MaxLength is the chat model's context window (Python
	// tools.chat_mdl.max_length). It bounds message_fit_in for prompt
	// assembly; <=0 falls back to chat.EffectiveContextLength's 8192 default.
	MaxLength int
	// ComposeAnswer enables the terminal composition node. Defaults to true
	// when a model is configured; set false to receive raw evidence only.
	ComposeAnswer *bool
	// Messages is the conversation history used by Formalize to resolve
	// pronouns/ellipses into a standalone question. The agentic and low graphs
	// formalize it as their first node (Python build_agentic_graph:803 and
	// build_low_graph:738); naive retrieval does not formalize.
	Messages []schema.Message
	// WebSearch is an optional provider for the `web_search` tool. When nil, the
	// tool is still advertised only if the mode enables it, but calls return a
	// clean MISS instead of falling through to the default branch (mirrors
	// Python's provider-gated web_search path).
	WebSearch harness.WebSearcher
	// DocIDVerifier checks which requested document ids really belong to the
	// datasets being searched (Python RAGTools._filter_known_doc_ids). Nil
	// disables the check and leaves the requested scope untouched.
	DocIDVerifier harness.DocIDVerifier
	// Cache answers near-identical re-asks from an earlier answer (Python
	// RAGTools._rag_cache). Nil disables the reuse. It is conversation-scoped:
	// the caller must pass the same *RAGCache on every call of a conversation.
	Cache *RAGCache
	// OriginalQuestion is the user's own, unrewritten question (Python
	// tools.original_user_question). When the outer caller passes a compressed
	// rewrite as Question, the original is preferred if both clearly describe
	// the same turn.
	OriginalQuestion string
	// TextAttachments is appended to the question and bypasses the re-ask cache
	// (Python tools.text_attachments_content).
	TextAttachments string
	// AnswerSink receives the answer as it is produced (Python
	// tools.answer_sink). Nil disables streaming; the answer is still returned
	// in full. Requires a model implementing harness.StreamingSessionModel,
	// otherwise the run falls back to one-shot composition.
	AnswerSink *AnswerSink
	// ToolStarted is called once research begins, before any retrieval, so the
	// caller can show progress (Python tools.tool_started_sink).
	ToolStarted func()
	// Retrieval tuning, mirroring Python RAGTools.retrieve: an explicit tool
	// argument still wins over these, and zero selects the harness defaults.
	TopN                int
	SimilarityThreshold float64
	// VectorSimilarityWeight is the vector leg's weight (Python
	// tools.vector_similarity_weight). A pointer so an explicit 0 — keyword-only,
	// which is what Python's agentic retrieve uses — is distinguishable from
	// "not configured". Nil selects DefaultAgenticVectorWeight (0).
	VectorSimilarityWeight *float64
	// UsingEmbedding is the Go spelling of Python RAGTools.retrieve's
	// `using_embedding: bool = False` (agentic_rag.py:599). When false (the
	// agentic default) retrieval is keyword-only (vector weight 0); when true
	// the embedder is engaged and VectorSimilarityWeight (default 0.7) applies.
	UsingEmbedding        bool
	RerankCandidatesCount int
	TopK                  int
	// DocScope is the session-wide document restriction (Python
	// tools.doc_scope). Nil means "search everything".
	DocScope []string
	// MetaDataFilter restricts retrieval by chunk metadata (Python
	// tools.meta_data_filter).
	MetaDataFilter map[string]any
	// KBs mirrors Python RAGTools' self.kbs (agentic_rag.py:283) — the full
	// Knowledgebase objects (carrying parser_config / tenant_id) the agentic
	// tools run over. Python loads these via
	// KnowledgebaseService.get_by_ids(kb_ids) in __init__; Go receives the
	// already-resolved objects from the caller (this package stays DB-free) and
	// feeds them to the Tagger, mirroring retrieve's
	// `rank_feature=label_question(question, self.kbs)`.
	KBs []*entity.Knowledgebase
	// Tagger mirrors Python RAGTools.retrieve's
	// `rank_feature=label_question(question, self.kbs)` (agentic_rag.py:668): it
	// classifies the query into question-type tags the retriever uses to boost
	// matching chunks. It is the Go equivalent of rag.app.tag.label_question,
	// implemented by internal/service.MetadataService.LabelQuestion. Nil means
	// no tag boost (Python's label_question returning None).
	Tagger harness.QuestionLabeler
	// Stats receives per-phase LLM usage for this run (Python
	// RAGTools.llm_stats, mirrored by harness.LLMUsageStats). Nil disables
	// collection.
	Stats *harness.LLMUsageStats
}

// sessionDeps projects RAGTools onto the harness.SessionDeps the action session
// and slot-table builders consume.
func (d RAGTools) sessionDeps() harness.SessionDeps {
	if d.Prompts == nil {
		return harness.SessionDeps{Model: d.Model}
	}
	return harness.SessionDeps{
		Tools:   d.Tools,
		Model:   d.Model,
		Prompts: d.Prompts,
		KB:      d.KB,
	}
}

func (d RAGTools) logger() *log.Logger {
	if d.Logger != nil {
		return d.Logger
	}
	return _LOG
}

// Question formalization (the graph's first node).
//
// Mirrors Python agentic_rag.RAGTools.formalize (line 412): rewrite the latest
// user message into a standalone question AND derive its search keywords, in one
// LLM call.
//
// Single-turn shortcut: when there is nothing to resolve, the question is kept
// VERBATIM (no rewrite) and only keywords are extracted. Rewriting a
// self-contained question risks silently changing its meaning — and the
// single-turn case is the overwhelming majority.
//
// The keyword/JSON extraction helpers (ExtractWeightedKeywords, ExtractJSON)
// stay in harness as the leaf capability library.

const (
	// formalizeTimeoutS bounds the formalize call.
	formalizeTimeoutS = 45.0
	// formalizeTemperature mirrors Python's chat conf at agentic_rag.py:471:
	// formalization is a mechanical rewrite, so it must be stable.
	formalizeTemperature = 0.1
)

var reFormalizeThink = regexp.MustCompile(`(?s)^.*</think>`)

var formalizePrompt = (`You are given a conversation. Do BOTH of the following and return JSON only:
1. Rewrite the LAST user message into a single, self-contained question that can be understood without the prior conversation — resolve pronouns, ellipses and follow-up shortcuts using the earlier turns. In most cases, it should be EXACTLY THE SAME as the last user query — only rewrite when there is something to resolve (a pronoun/ellipsis pointing back at an earlier turn). Preserve the original language.
2. Extract keywords for a keyword search: the salient content words and phrases that literally appear in the (standalone) question — key nouns, named entities, domain terms — PLUS 2-3 close synonyms/abbreviations/aliases/alternative spellings of each, in the SAME language as the question. Maximize recall. Do NOT include terms that would be part of the answer.
   Example — "In which year did Apple acquire Beats?" -> keywords = "Apple, Apple Inc., AAPL, acquire, acquisition, acquired, Beats, Beats Electronics".

Output ONLY JSON, no prose, no code fences: {"question": "<standalone question>", "keywords": "<term1, term2, synonym1, ...>"}`)

// Formalize mirrors Python RAGTools.formalize: return (question, keywords) for
// the given conversation.
//
// messages may be []schema.Message (preferred) or pre-formatted "Speaker: text"
// strings. On any failure it degrades to (last user message, "") — formalization
// is an optimization, never a precondition for answering.
//
// maxLength is the chat model's context window (Python tools.chat_mdl.max_length);
// it bounds the prompt fit. <=0 falls back to chat.EffectiveContextLength's 8192.
func Formalize(ctx context.Context, deps harness.SessionDeps, messages []schema.Message, maxLength int) (string, string) {
	lastUser, transcript := transcriptOf(messages)
	if lastUser == "" {
		return "", ""
	}
	if deps.Model == nil {
		// No model: keep the raw question and skip keyword extraction rather
		// than failing the request.
		return lastUser, ""
	}

	// Single-turn: nothing to resolve — keep the question VERBATIM (rewriting a
	// self-contained question risks silently changing its meaning), and extract
	// only the search keywords (Python: tools.extract_keywords).
	if !isMultiTurn(messages) {
		_LOG.Printf("[Formalize] Single-turn self-contained question — kept verbatim (no rewrite): %s", trunc(lastUser, 120))
		_, kw := harness.ExtractWeightedKeywords(ctx, deps.Model, lastUser)
		return lastUser, kw
	}

	ctx, done := harness.Phase(ctx, harness.PhaseFormalize)
	// Phase returns a child context carrying the phase label; the timeout is
	// derived from it so the call is both attributed and bounded.
	defer done()

	callCtx, cancel := context.WithTimeout(ctx, deadlineToDuration(formalizeTimeoutS))
	defer cancel()

	// Python 470: message_fit_in(form_message(system, user), chat_mdl.max_length).
	fitted, fitErr := chat.FitMessages(formalizePrompt, []schema.Message{
		*schema.UserMessage("Conversation:\n" + transcript + "\n\nOutput JSON:"),
	}, maxLength)
	if fitErr != "" {
		_LOG.Printf("[Formalize] prompt fitting failed: %s", fitErr)
		return lastUser, ""
	}
	system := formalizePrompt
	history := fitted
	if len(fitted) > 0 && fitted[0].Role == schema.System {
		system = fitted[0].Content
		history = fitted[1:]
	}
	msgs := make([]schema.Message, 0, 1+len(history))
	msgs = append(msgs, *schema.SystemMessage(system))
	msgs = append(msgs, history...)

	// Python 471: async_chat(system, history, {"temperature": 0.1}).
	reply, err := modelWithTemperature(deps.Model, formalizeTemperature).Complete(callCtx, msgs, nil)
	if err != nil {
		_LOG.Printf("[Formalize] failed; keeping the raw question: %v", err)
		return lastUser, ""
	}

	data, _ := harness.ExtractJSON(stripThinkAndFences(reply.Content)).(map[string]any)
	question := ""
	if data != nil {
		if raw, ok := data["question"]; ok && raw != nil {
			question = strings.Trim(strings.TrimSpace(fmt.Sprint(raw)), "\"'")
		}
	}
	if question == "" {
		// Fall back to the raw last user message rather than an empty question.
		question = lastUser
	}
	keywords := ""
	if data != nil {
		switch v := data["keywords"].(type) {
		case string:
			keywords = v
		case []any:
			parts := make([]string, 0, len(v))
			for _, k := range v {
				if s := strings.TrimSpace(fmt.Sprint(k)); s != "" {
					parts = append(parts, s)
				}
			}
			keywords = strings.Join(parts, ", ")
		}
	}
	return question, harness.CompactKeywords(keywords)
}

// transcriptOf renders the conversation and extracts the last user message.
func transcriptOf(messages []schema.Message) (lastUser, transcript string) {
	var lines []string
	for _, m := range messages {
		content := m.Content
		switch m.Role {
		case schema.User:
			lastUser = strings.TrimSpace(content)
			lines = append(lines, "User: "+content)
		case schema.Assistant:
			lines = append(lines, "Assistant: "+content)
		case schema.System:
			lines = append(lines, "System: "+content)
		default:
			lines = append(lines, strings.TrimSpace(content))
		}
	}
	return strings.TrimSpace(lastUser), strings.Join(lines, "\n")
}

// isMultiTurn reports whether the conversation has more than one user turn.
// Mirrors Python's `multi_turn = len(user_msgs) > 1`.
func isMultiTurn(messages []schema.Message) bool {
	n := 0
	for _, m := range messages {
		if m.Role == schema.User {
			n++
		}
	}
	return n > 1
}

// stripThinkAndFences mirrors Python agentic_rag.py:474-475: drop a leading
// thinking preamble, then strip Markdown fences. Python's fence regex removes
// the delimiters wherever they appear, so a truncated or inline fence leaves no
// residue either.
func stripThinkAndFences(s string) string {
	s = reFormalizeThink.ReplaceAllString(s, "")
	s = reFenceDelimiters.ReplaceAllString(s, "")
	return strings.TrimSpace(s)
}

// reFenceDelimiters mirrors Python's `re.sub(r"```(?:json)?\s*|\s*```", "", s)`.
var reFenceDelimiters = regexp.MustCompile("```(?:json)?\\s*|\\s*```")

// End-to-end entry point: one call that runs the harness and publishes the
// evidence it collected.
//
// Rag mirrors Python RAGTools.rag (agentic_rag.py:817) — the RAGTools method
// that drives the research pipeline. It owns the conversation-level concerns
// (re-ask cache, the effective question, attachments, composing the final
// answer) and delegates the mode dispatch to RunAgenticRAG, which mirrors
// run_agentic_rag (agentic_rag_graph.py:1571) and carries the dispatch of
// :1581:
//
//	mode is NAIVE      → _naive_rag            (plain retrieval)
//	mode.Agentic       → build_agentic_graph   (medium / high / ultra)
//	else               → build_low_graph       (low: formalize → direct_search)
//
// SCOPE NOTE — read before relying on this for agentic modes: the outer
// orchestration that agentic_rag_graph.py owns — planner decomposition, prefetch
// fan-out, and the SCA↔rewriter iteration loop that drives medium/high/ultra — is
// the advanced_rag package's agentic planner (Rag's agentic-loop registration).
// So:
//
//   - low / naive: fully wired here (direct search through HybridSearch).
//   - medium/high/ultra: Rag delegates to the registered AgenticLoop, which the
//     advanced_rag package registers (see SetAgenticLoop). Without it, Rag falls
//     back to a single action session.
//
// The harness package keeps the low-level retrieval / session / tool machinery
// (RunActionSession, HybridSearch, Toolset, Kbinfos, searchExecutor, ...).

// RunRequest lives in the harness package (see harness/tool_executor.go) because the
// harness engine's searchExecutor needs it to carry the per-run query, and the
// advanced_rag package must not be imported from harness (import-cycle rule).
// RunResponse is the agent-side result type returned by RAGTools.rag.

// RunResponse is the outcome of one harness run.
type RunResponse struct {
	// Answer is the session's terminal answer, when it produced one.
	Answer string
	// Slots are the filled slot-table values discovered by the session.
	Slots []harness.Variable
	// Chunks is the evidence accumulated in Kbinfos (narrowed, as the model saw
	// it). Citations are built from this.
	Chunks []map[string]any
	// DocAggs is the per-document aggregation of Chunks.
	DocAggs []map[string]any
	// EmptyResult is true when nothing was retrieved, which the caller turns
	// into an "I don't have enough information" answer (Python direct_search's
	// empty_result).
	EmptyResult bool
	// Mode is the resolved spec, for logging.
	Mode harness.ModeSpec
	// Partial is true when research ended without a satisfying verdict, so the
	// caller surfaces the residual findings honestly instead of refusing.
	Partial bool
	// SearchRounds is the number of completed SCA→rewrite iterations (0 for the
	// non-agentic paths).
	SearchRounds int
	// GraphFailed is true when the research graph itself errored. Python pairs it
	// with "produced nothing" before falling back to an internal-error message
	// (:1667-1668); an empty result on its own is not a failure.
	GraphFailed bool
	// Verdict is the final sufficiency verdict ("SUFFICIENT"/"INSUFFICIENT").
	Verdict string
	// SCAFeedback is the body of the SCA feedback note (status hint + hard
	// violations + missing claims + confidence) — agentic_rag.py:878 / :902-921's
	// string built from sca.to_grounded(). Rag() appends it as the "[Research
	// status]" note for every INSUFFICIENT verdict, adding the trailing "STOP" vs
	// "call rag again" sentence based on the consecutive-unanswerable count.
	SCAFeedback string
	// CollectedAnswer is the research draft (SCA-reviewed) produced by the
	// agentic loop. It feeds the final composition; prefer Answer for display.
	CollectedAnswer string
	// Kbinfos carries the full accumulated state (including the lossless
	// memory store) for callers that need more than the summary above.
	Kbinfos *harness.Kbinfos
}

// RAGTools is the single run-dependency object. Construct it directly (RAGTools{
// ... }); its zero value is a valid starting point for the narrower call sites
// that only set a subset of fields.

// AnswerSink forwards a partially produced answer while the model is still
// writing it (Python tools.answer_sink). Nil disables streaming; the answer is
// still returned in full.
//
// Requires a model implementing harness.StreamingSessionModel; otherwise the run
// falls back to a single completion.
type AnswerSink struct {
	// OnDelta receives each successive piece of the answer. isThink marks pieces
	// of a hidden reasoning block, which must not be shown as part of the answer
	// (Python answer_sink(delta, kind == "think")).
	OnDelta func(delta string, isThink bool)
	// OnReset drops what has been forwarded so far. It is called before a
	// fallback re-sends the answer from scratch, so a partially streamed answer
	// is never shown twice. Nil means the sink cannot rewind, and the caller
	// only sends the complete answer.
	OnReset func()
}

// deliver forwards one piece, if the sink has a handler.
func (s *AnswerSink) deliver(delta string, isThink bool) {
	if s == nil || s.OnDelta == nil || delta == "" {
		return
	}
	s.OnDelta(delta, isThink)
}

// reset discards previously forwarded pieces, if the sink supports it.
func (s *AnswerSink) reset() {
	if s == nil || s.OnReset == nil {
		return
	}
	s.OnReset()
}

// KeywordExtractorFn extracts the weighted retrieval query for a question.
type KeywordExtractorFn func(ctx context.Context, question string) (retrievalQuery string, err error)

// Run executes one harness request end to end and publishes the evidence into
// the canvas state (when the context carries one) so citation grounding can
// read it.
//
// It never fails for a recoverable reason: a missing component degrades to
// "no evidence" rather than erroring, matching Python's behaviour of logging
// and returning an empty kbinfos.
// ragCacheMinOverlap is the word-overlap ratio at which a new question counts
// as a re-ask of a cached one (Python _RAG_CACHE_MIN_OVERLAP).
const ragCacheMinOverlap = 0.6

// ragCacheMinShared is the minimum number of shared significant words before a
// cached answer may be reused (Python _RAG_CACHE_MIN_SHARED).
const ragCacheMinShared = 2

// effectiveQuestionMinShared is the overlap _resolve_effective_question needs
// before trusting the original question over the outer rewrite. Python uses the
// literal 2 here, independent of _RAG_CACHE_MIN_SHARED.
const effectiveQuestionMinShared = 2

// ragCacheStopwords is Python _RAG_CACHE_STOPWORDS: for cross-`rag`-call dedup
// only, never for retrieval or answer quality.
var ragCacheStopwords = map[string]bool{
	"the": true, "a": true, "an": true, "is": true, "was": true, "were": true,
	"what": true, "which": true, "when": true, "where": true, "who": true,
	"how": true, "of": true, "in": true, "to": true, "for": true, "and": true,
	"or": true, "but": true, "on": true, "at": true, "by": true, "be": true,
	"as": true, "it": true, "that": true, "this": true, "about": true,
	"with": true, "their": true, "its": true, "have": true, "has": true,
	"had": true, "been": true, "being": true, "from": true, "over": true,
	"under": true, "do": true, "does": true, "did": true, "not": true,
	"no": true, "yes": true, "can": true, "could": true, "should": true,
	"would": true, "also": true, "only": true, "very": true, "much": true,
	"more": true, "most": true, "some": true, "any": true,
}

// reQuestionTokens mirrors Python's `re.findall(r"[a-zA-Z0-9一-鿿]+", ...)`.
var reQuestionTokens = regexp.MustCompile(`[a-zA-Z0-9\x{4e00}-\x{9fff}]+`)

// questionGram is Python _question_keywords' return value: the significant
// words plus the numeric tokens kept apart, so questions naming different
// numbers are never treated as the same question.
type questionGram struct {
	words   map[string]bool
	numbers map[string]bool
}

// questionKeywords mirrors Python _question_keywords. For English, plain
// tokenisation suffices; CJK tokens survive as whole significant units.
func questionKeywords(question string) questionGram {
	gram := questionGram{words: map[string]bool{}, numbers: map[string]bool{}}
	tokens := reQuestionTokens.FindAllString(strings.ToLower(question), -1)
	for _, t := range tokens {
		if isDigitToken(t) {
			gram.numbers[t] = true
		}
	}
	for _, t := range tokens {
		if len([]rune(t)) > 1 && !isDigitToken(t) && !ragCacheStopwords[t] {
			gram.words[t] = true
		}
	}
	if len(gram.words) == 0 {
		// Python falls back to every non-numeric token, so an all-stopword
		// question still has something to compare.
		for _, t := range tokens {
			if len([]rune(t)) > 1 && !isDigitToken(t) {
				gram.words[t] = true
			}
		}
	}
	return gram
}

func isDigitToken(s string) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		if r < '0' || r > '9' {
			return false
		}
	}
	return true
}

// cacheSimilar mirrors Python _cache_similar: significant-word overlap
// (shared / min cardinality), and numbers must be both empty or identical.
func cacheSimilar(a, b questionGram) bool {
	if len(a.words) == 0 || len(b.words) == 0 {
		return false
	}
	if len(a.numbers) > 0 || len(b.numbers) > 0 {
		if !sameTokens(a.numbers, b.numbers) {
			return false
		}
	}
	shared := 0
	for w := range a.words {
		if b.words[w] {
			shared++
		}
	}
	if shared < ragCacheMinShared {
		return false
	}
	minLen := len(a.words)
	if len(b.words) < minLen {
		minLen = len(b.words)
	}
	return float64(shared)/float64(minLen) >= ragCacheMinOverlap
}

func sameTokens(a, b map[string]bool) bool {
	if len(a) != len(b) {
		return false
	}
	for k := range a {
		if !b[k] {
			return false
		}
	}
	return true
}

// FitEvidence mirrors Python RAGTools._fit_evidence (agentic_rag.py:716): trim
// evidence so question + evidence + the template stay inside a FIXED budget.
//
// The budget is deliberately not the model's full context: a large retrieval
// pool must never fill the window with evidence, so each sufficiency/answer call
// stays cheap. message_fit_in keeps the small side (the question) whole and
// trims the large side (the evidence), which is what the two-message shape
// below asks for.
func FitEvidence(question, evidence string) string {
	if evidence == "" {
		return evidence
	}
	fitted, fitErr := chat.FitMessages("", []schema.Message{
		*schema.UserMessage(question),
		*schema.UserMessage(evidence),
	}, evidenceBudgetTokens)
	if fitErr != "" || len(fitted) == 0 {
		return evidence
	}
	return fitted[len(fitted)-1].Content
}

// routerPromptBody is Python's router_prompt (agentic_rag.py:383-403) with the
// summarize_document line left as %s: that line is only included when
// unstructured retrieval is available (Python has_unstructured).
const routerPromptBody = "You are a smart agent. For any question that needs " +
	"evidence from the knowledge bases or the web, call the `rag` tool " +
	"with a self-contained question — it runs the full search-and-answer " +
	"pipeline and returns a cited answer.\n" +
	"After the `rag` tool returns, do not call `rag` again for the same " +
	"user question. Use the returned cited answer as the final answer " +
	"unless the user explicitly asks a new question.\n" +
	"CRITICAL — preserve the full multi-hop structure when phrasing the " +
	"`rag` question. A question that compares two or more DISTINCT " +
	"targets or needs an arithmetic result across them (\"how much taller " +
	"is X than Y\", \"how many days after A's death did B die\", \"which of " +
	"these was discovered last\") MUST keep every target and relation in " +
	"the question you pass to `rag`. Never rewrite a comparison into a " +
	"single-entity question — dropping the second entity (e.g. the " +
	"purchaser in \"how many days after his death did the man who " +
	"purchased it in 1933 die\") makes the pipeline answer only the first " +
	"part. Pass the complete comparison.\n" +
	"%s" +
	"Do not invent facts and do not fabricate document IDs."

// routerSummarizeLine is the conditional summarize_document instruction
// (Python agentic_rag.py:378-381).
const routerSummarizeLine = "- Call `summarize_document` ONLY when the user explicitly asks to summarise a specific document ('summarise the security audit', 'tldr the onboarding guide'). It needs a document ID.\n"

// SysPrompt mirrors Python RAGTools.sys_prompt (agentic_rag.py:371): the thin
// router prompt for callers that bind the tool set. The workflow itself lives in
// the rag graph; the outer model only chooses between retrieval and an explicit
// single-document summary.
//
// hasUnstructured mirrors Python has_unstructured(): when false the
// summarize_document instruction is omitted. systemPrompt is the dialog-level UI
// configuration, prepended when set (Python :404-405).
func SysPrompt(systemPrompt string, hasUnstructured bool) string {
	summarizeLine := ""
	if hasUnstructured {
		summarizeLine = routerSummarizeLine
	}
	router := fmt.Sprintf(routerPromptBody, summarizeLine)
	if strings.TrimSpace(systemPrompt) != "" {
		return systemPrompt + "\n\n" + router
	}
	return router
}

// GetCitationGuidelines mirrors Python RAGTools.get_citation_guidelines
// (agentic_rag.py:367): the citation rules the final answer must follow, with
// an optional user-defined override.
//
// Python renders `citation_prompt(self.user_defined_prompts)`; Go loads the same
// citation_prompt.md via prompts.CitationPrompt and takes a non-empty override
// verbatim — exactly as Python returns citation_prompt(self.user_defined_prompts).
func GetCitationGuidelines(userDefined string) string {
	return prompts.CitationPrompt(userDefined)
}

// resolveEffectiveQuestion mirrors Python _resolve_effective_question: prefer
// the user's ORIGINAL, complete question over the outer model's rewrite, but
// only when both clearly describe the same user turn. The outer rewrite often
// drops the final target of a multi-hop question, and no later stage can
// recover a deleted answer-attribute.
func resolveEffectiveQuestion(question, originalUserQuestion string) string {
	if question == "" || originalUserQuestion == "" {
		return question
	}
	original := strings.TrimSpace(originalUserQuestion)
	if original == "" {
		return question
	}
	qk := questionKeywords(question)
	ok := questionKeywords(original)
	if len(qk.words) == 0 || len(ok.words) == 0 {
		return question
	}
	shared := 0
	for w := range qk.words {
		if ok.words[w] {
			shared++
		}
	}
	if shared >= effectiveQuestionMinShared {
		return original
	}
	return question
}

// RAGCache mirrors Python RAGTools._rag_cache (agentic_rag.py:837-889): it
// answers a near-identical re-ask from a previous answer instead of re-running
// the whole graph.
//
// Lifetime: Python stores it on the RAGTools instance, which lives for one
// dialog turn, so the caller MUST hold one *RAGCache per conversation and pass
// it on every Run call. Building a fresh cache per call disables the reuse.
// Nil disables it outright.
type RAGCache struct {
	mu          sync.Mutex
	entries     map[string]ragCacheEntry
	lastVerdict string
	// ConsecutiveUnanswerable mirrors Python RAGTools._consecutive_unanswerable
	// (agentic_rag.py:818): how many consecutive rag() calls ended without a
	// satisfying verdict. After two in a row, RAGTools.rag appends a
	// "[Research status] … STOP calling rag again" note to the answer so the
	// outer agent stops re-asking. It lives on the conversation-scoped cache so
	// it survives across turns, exactly like Python keeps it on the RAGTools
	// instance (which itself lives one dialog turn).
	ConsecutiveUnanswerable int
}

type ragCacheEntry struct {
	answer string
	gram   questionGram
}

// NewRAGCache returns an empty conversation-scoped cache.
func NewRAGCache() *RAGCache {
	return &RAGCache{entries: map[string]ragCacheEntry{}}
}

// Lookup returns the cached answer for a question judged near-identical to an
// earlier one. Attachments bypass the cache (their content is not part of the
// key), and a previous round that was not SUFFICIENT invalidates reuse: the
// caller asked again precisely because it needs more evidence.
func (c *RAGCache) Lookup(question string) (string, bool) {
	if c == nil || question == "" || !c.reuseAllowed() {
		return "", false
	}
	gram := questionKeywords(question)

	c.mu.Lock()
	defer c.mu.Unlock()
	for _, e := range c.entries {
		if cacheSimilar(gram, e.gram) {
			return e.answer, true
		}
	}
	return "", false
}

// Store records answer for question.
func (c *RAGCache) Store(question, answer string) {
	if c == nil || question == "" || answer == "" {
		return
	}
	gram := questionKeywords(question)
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.entries == nil {
		c.entries = map[string]ragCacheEntry{}
	}
	c.entries[question] = ragCacheEntry{answer: answer, gram: gram}
}

// sessionCaches holds one near-duplicate answer cache per conversation, so
// callers that build RAGTools per call still get conversation-scoped reuse
// instead of a fresh, useless cache. Python gets this for free because
// RAGTools._rag_cache lives on an instance that spans the dialog.
var sessionCaches = struct {
	mu      sync.Mutex
	byID    map[string]*RAGCache
	lastUse map[string]time.Time
}{byID: map[string]*RAGCache{}, lastUse: map[string]time.Time{}}

// sessionCacheMaxIdle is how long an unused conversation cache is kept. Without
// an eviction the map would grow with every distinct conversation.
const sessionCacheMaxIdle = 30 * time.Minute

// SessionCache returns the conversation-scoped cache for sessionID, creating it
// on first use. An empty sessionID gets a throwaway cache: scoping reuse by
// nothing would let one conversation answer for another.
func SessionCache(sessionID string) *RAGCache {
	if sessionID == "" {
		return NewRAGCache()
	}
	now := time.Now()

	sessionCaches.mu.Lock()
	defer sessionCaches.mu.Unlock()
	for id, used := range sessionCaches.lastUse {
		if now.Sub(used) > sessionCacheMaxIdle {
			delete(sessionCaches.byID, id)
			delete(sessionCaches.lastUse, id)
		}
	}
	c := sessionCaches.byID[sessionID]
	if c == nil {
		c = NewRAGCache()
		sessionCaches.byID[sessionID] = c
	}
	sessionCaches.lastUse[sessionID] = now
	return c
}

// noteVerdict records the verdict of the last research round.
func (c *RAGCache) noteVerdict(verdict string) {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.lastVerdict = verdict
}

// reuseAllowed mirrors Python's `_cache_ok = not last_status or last_status ==
// "SUFFICIENT"`.
func (c *RAGCache) reuseAllowed() bool {
	if c == nil {
		return false
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.lastVerdict == "" || c.lastVerdict == "SUFFICIENT"
}

func Rag(ctx context.Context, deps RAGTools, req harness.RunRequest) *RunResponse {
	logger := deps.Logger
	if logger == nil {
		logger = _LOG
	}
	spec := harness.GetMode(req.ThinkingMode)
	if !IsKnownMode(req.ThinkingMode) {
		// An unrecognised label resolves to NAIVE; log it so a typo in the mode
		// is visible instead of silently downgrading the request.
		logger.Printf("[Agentic RAG] unrecognised thinking mode %q; falling back to naive (non-agentic) retrieval", req.ThinkingMode)
	}

	// Python :831 — tell the caller research is starting, before any retrieval
	// work, so it can show progress for the (potentially long) graph run.
	if deps.ToolStarted != nil {
		deps.ToolStarted()
	}

	// Re-ask guard: reuse a near-identical question's cached answer instead of
	// re-running the graph (Python rag:837-858). Attachments bypass it, since
	// their content is appended to the question below and is not part of the key.
	cache := deps.Cache
	if cache == nil && req.SessionID != "" {
		// Callers that build RAGTools per call get the conversation's cache, so
		// reuse works without them holding any state.
		cache = SessionCache(req.SessionID)
	}
	cacheable := deps.TextAttachments == ""
	if cacheable && cache != nil {
		if cached, hit := cache.Lookup(req.Question); hit {
			logger.Printf("[Agentic RAG] cache hit — reused prior answer for near-identical question %q; skipped research", trunc(req.Question, 80))
			return &RunResponse{Answer: cached, Mode: spec}
		}
	}

	// Prefer the user's ORIGINAL, complete question over the outer rewrite
	// (Python rag:865). The outer rewrite often drops the final target of a
	// multi-hop question, and no later stage can recover it.
	if deps.OriginalQuestion != "" {
		if effective := resolveEffectiveQuestion(req.Question, deps.OriginalQuestion); effective != req.Question {
			logger.Printf("[Agentic RAG] using original user question over outer rewrite (original=%q → rewrite=%q)",
				trunc(deps.OriginalQuestion, 80), trunc(req.Question, 80))
			req.Question = effective
		}
	}
	if deps.TextAttachments != "" {
		req.Question += deps.TextAttachments
	}

	tenantID := req.TenantID
	if tenantID == "" {
		tenantID = harness.TenantIDFromContext(ctx)
	}
	datasetIDs := req.DatasetIDs

	retriever := deps.Retriever
	if retriever == nil {
		retriever = &harness.RuntimeRetriever{}
	}

	kb := &harness.Kbinfos{}
	searchDeps := harness.SearchDeps{
		Backend:       retriever,
		KbIDs:         datasetIDs,
		TenantID:      tenantID,
		KB:            kb,
		Expand:        deps.Expand,
		Logger:        logger,
		DocIDVerifier: deps.DocIDVerifier,
		Model:         deps.Model, // the calculate tool writes its expression via the model
		DocScope:      deps.DocScope,
		// Python retrieve:614-646 — configuration is the middle precedence
		// level, between an explicit tool argument and the module defaults.
		TopN:                   deps.TopN,
		SimilarityThreshold:    deps.SimilarityThreshold,
		VectorSimilarityWeight: deps.VectorSimilarityWeight,
		UsingEmbedding:         deps.UsingEmbedding,
		RerankCandidatesCount:  deps.RerankCandidatesCount,
		TopK:                   deps.TopK,
		MetaDataFilter:         deps.MetaDataFilter,
		// rank_feature (Python retrieve:668): RAGTools carries the KB objects
		// and a tagger, mirroring rank_feature=label_question(question, self.kbs).
		KBs:    deps.KBs,
		Tagger: deps.Tagger,
		// External embedding handle (Python tools.embed_mdl). When the caller
		// supplied one on RAGTools it is used directly; nil keeps the package's
		// internal tenant-default resolver as a fallback (which itself degrades
		// to keyword matching when the DB is uninitialised or encoding panics).
		Embedder: deps.Embedder,
	}

	resp := &RunResponse{Mode: spec, Kbinfos: kb}

	// Python :877 — hand over to the graph driver, which owns the mode dispatch
	// (agentic graph vs. direct/naive retrieval) and formalization.
	RunAgenticRAG(ctx, deps, req, searchDeps, kb, resp, logger, spec)

	resp.Chunks = kb.Chunks
	resp.DocAggs = kb.DocAggs
	resp.EmptyResult = len(kb.Chunks) == 0

	// Publish evidence for citation grounding. Best-effort: when no canvas
	// state is attached (e.g. unit tests), skip silently — mirroring
	// RetrievalTool's behaviour.
	if len(kb.Chunks) > 0 {
		harness.PublishReferences(ctx, kb)
	}

	// Compose the final grounded answer — the graph's last node, shared by every
	// thinking mode (Python: _compose_answer_from_evidence / _naive_rag).
	composeFinalAnswer(ctx, deps, req, kb, resp, logger)

	// Python rag (:902-929) appends a "[Research status]" note for EVERY
	// non-SUFFICIENT verdict. When research has stayed unsatisfying for two
	// consecutive turns it tells the outer agent to STOP calling rag again;
	// otherwise it invites a focused re-ask. The counter lives on the
	// conversation-scoped cache; it was incremented back in NewAgenticLoop once
	// the SCA verdict was known. Skip the naive/sufficient paths: an empty
	// answer or a SUFFICIENT verdict has nothing to annotate.
	if resp.Verdict == VerdictInsufficient && resp.SCAFeedback != "" && resp.Answer != "" {
		var trailing string
		if deps.Cache != nil && deps.Cache.ConsecutiveUnanswerable >= 2 {
			trailing = " STOP calling rag again: the same gaps remain."
		} else {
			trailing = " If these gaps are material, call rag again with a question focused on them."
		}
		resp.Answer += "\n\n[Research status] " + resp.SCAFeedback + trailing
	}

	// Cache the freshly produced answer for later near-identical questions, and
	// remember the verdict so a following re-ask is not answered from an
	// admittedly incomplete one (Python rag:888-889 and :848-852).
	if cacheable && cache != nil {
		cache.Store(req.Question, resp.Answer)
		cache.noteVerdict(resp.Verdict)
	}
	return resp
}

// composeFinalAnswer runs the graph's terminal node and writes the result onto
// the response.
//
// When no model is configured this is a no-op: the caller still receives the
// evidence and can answer with it, matching Python's behaviour of returning an
// empty kbinfos rather than failing.
func composeFinalAnswer(ctx context.Context, deps RAGTools, req harness.RunRequest, kb *harness.Kbinfos, resp *RunResponse, logger *log.Logger) {
	if deps.Model == nil {
		return
	}
	if deps.ComposeAnswer != nil && !*deps.ComposeAnswer {
		return
	}
	adeps := AnswerDeps{
		Model:         deps.Model,
		CiteRules:     deps.CiteRules,
		SystemPrompt:  deps.SystemPrompt,
		EmptyResponse: deps.EmptyResponse,
		MaxTokens:     deps.EvidenceMaxTokens,
		MaxLength:     deps.MaxLength,
		Logger:        logger,
	}
	// Stream the answer when the model and the caller both support it, so the
	// user sees text while it is produced instead of only at the end.
	if deps.AnswerSink != nil {
		if streamer, ok := deps.Model.(harness.StreamingSessionModel); ok {
			deps.AnswerSink.reset()
			streamed, err := ComposeAnswerStream(ctx, adeps, streamer, kb, req.Question, resp.Partial, func(delta string, isThink bool) error {
				deps.AnswerSink.deliver(delta, isThink)
				return nil
			})
			if err == nil {
				resp.Answer = streamed.Answer
				if streamed.Partial {
					resp.Partial = true
				}
				return
			}
			// A partially streamed answer must not be sent twice: tell the sink
			// to drop what it already forwarded and fall back to one shot.
			logger.Printf("[Agentic RAG] streaming compose failed (%v); falling back to a single call", err)
			deps.AnswerSink.reset()
		}
	}
	if deps.Stats != nil {
		deps.Stats.RecordCall("compose")
	}
	res := ComposeAnswer(ctx, adeps, kb, req.Question, resp.Partial, false)
	if deps.Stats != nil && res.Failed {
		deps.Stats.RecordFailed("compose")
	}
	resp.Answer = res.Answer
	if res.Partial {
		resp.Partial = true
	}
	if deps.AnswerSink != nil {
		deps.AnswerSink.deliver(res.Answer, false)
	}
}

// runDirect is the low/naive path: one hybrid search, no tool loop.
func runDirect(ctx context.Context, deps RAGTools, req harness.RunRequest, sd harness.SearchDeps, kb *harness.Kbinfos, resp *RunResponse, logger *log.Logger) {
	ctx, done := harness.Phase(ctx, harness.PhaseDirect)
	retrievalQuery := ""
	if deps.Keywords != nil {
		rq, err := deps.Keywords(ctx, req.Question)
		if err != nil {
			logger.Printf("[Agentic RAG] weighted keyword extraction failed: %v", err)
		} else {
			retrievalQuery = rq
		}
	} else if deps.Model != nil {
		// Default: the four-aspect weighted extraction (Python
		// tools.extract_keywords / harness/keywords.py). It gives BM25 both the
		// discriminating entity and the surface variants the corpus may use.
		rq, kw := harness.ExtractWeightedKeywords(ctx, deps.Model, req.Question)
		retrievalQuery = rq
		if req.Keywords == "" {
			req.Keywords = kw
		}
	}
	defer done()

	chunks, aggs := harness.HybridSearch(ctx, sd, harness.SearchParams{
		Question:       req.Question,
		Keywords:       req.Keywords,
		RetrievalQuery: retrievalQuery,
		UseCompiled:    req.UseCompiled,
		TopN:           req.TopN,
	})
	kb.Merge(chunks, aggs)
	if !kb.HasChunks() {
		logger.Printf("[Agentic RAG] direct search found no matching passages")
	}
}

// AgenticLoop runs the full agentic search loop (planner / prefetch / SCA↔
// rewriter iteration) for an agentic mode.
//
// It is registered by the advanced_rag package's init rather than called directly: the
// advanced_rag package owns RAGTools, so the harness cannot import it back. The
// indirection keeps the dependency graph acyclic. Registering is optional — see
// Run for the fallback when nothing is registered.
type AgenticLoop func(ctx context.Context, deps RAGTools, req harness.RunRequest, kb *harness.Kbinfos, resp *RunResponse, logger *log.Logger)

var agenticLoop AgenticLoop

// SetAgenticLoop registers the outer agentic loop. Called from the agent
// package's init(); importing that package (even blank) activates the full
// medium/high/ultra pipeline. Without it those modes degrade to a single action
// session.
func SetAgenticLoop(fn AgenticLoop) { agenticLoop = fn }

// runAgentic drives medium/high/ultra.
//
// When an agentic loop is registered (see SetAgenticLoop) it runs the full
// five-phase pipeline: planner fan-out → slot table → slot research rounds →
// SCA review → gap rewrite → repeat until sufficient or the budget runs out.
//
// Fallback (no loop registered): ONE action session via RunActionSession. This
// keeps `Run` usable without the planner, at the cost of the
// planner/fan-out/SCA iteration.
func runAgentic(ctx context.Context, deps RAGTools, req harness.RunRequest, sd harness.SearchDeps, kb *harness.Kbinfos, resp *RunResponse, logger *log.Logger) {
	// Formalization happens inside the graph: Python wires it as the
	// build_agentic_graph entry node (add_edge(START, "formalize_question")).
	if agenticLoop != nil {
		agenticLoop(ctx, deps, req, kb, resp, logger)
		return
	}
	if deps.Model == nil {
		logger.Printf("[Agentic RAG] no model configured for mode %q; degrading to a direct search", resp.Mode.Label)
		runDirect(ctx, deps, req, sd, kb, resp, logger)
		return
	}
	runSingleSession(ctx, deps, req, sd, kb, resp, logger)
}

// runSingleSession is the fallback agentic path: seed a slot table and run ONE
// bounded action session against it.
func runSingleSession(ctx context.Context, deps RAGTools, req harness.RunRequest, sd harness.SearchDeps, kb *harness.Kbinfos, resp *RunResponse, logger *log.Logger) {
	// Seed the slot table and run one bounded session against it. A fresh tool
	// cache and query history per run: they exist to dedupe WITHIN a session,
	// not across requests.
	sd.WebSearch = deps.WebSearch
	sd.CiteRules = deps.CiteRules
	sessionDeps := harness.SessionDeps{
		Tools: &harness.Toolset{
			ThinkingMode:  resp.Mode.Label,
			HasWebSearch:  resp.Mode.HasTool("web_search"),
			DisabledTools: map[string]bool{},
			Exec:          harness.NewSearchExecutor(sd, req),
		},
		Model:   deps.Model,
		Prompts: deps.Prompts,
		// Surface already-retrieved evidence into the action session seed so the
		// ReAct loop fills slots from what it has instead of re-searching.
		KB: kb,
	}

	// Python :1622-1624 — there is no whole-graph wall clock: research stays
	// bounded by per-node timeouts, the routing guards and the recursion limit.
	// The only budget is the one AgenticState carries (Python :823), and it is
	// set when the state is created, so formalization is not charged to it.
	init := harness.InitializeState(ctx, sessionDeps, req.Question, nil, req.DeadlineLeft)
	root := init.Root

	result := harness.RunActionSession(ctx, sessionDeps, req.Question, root, req.DeadlineLeft, "", nil, nil)
	// Python :1625-1629 — failure to run research is recorded separately from
	// "research found nothing": the session yielded neither messages nor states,
	// so nothing was produced at all.
	if len(result.Messages) == 0 && len(result.NewStates) == 0 && result.FoundAnswer == nil {
		resp.GraphFailed = true
		logger.Printf("[Agentic RAG] graph execution produced nothing (mode=%q)", resp.Mode.Label)
	}

	if result.FoundAnswer != nil {
		resp.Answer = *result.FoundAnswer
	}
	// Report the deepest slot table the session produced: later states carry
	// strictly more filled slots than the root.
	if len(result.NewStates) > 0 {
		best := result.NewStates[len(result.NewStates)-1]
		resp.Slots = append(resp.Slots, best.State...)
	} else {
		resp.Slots = append(resp.Slots, root.State...)
	}
}

// IsKnownMode reports whether label names a configured thinking mode.
func IsKnownMode(label string) bool {
	_, ok := harness.THINKING_MODES[strings.ToLower(strings.TrimSpace(label))]
	return ok
}

// DocIDLookup mirrors Python RAGTools._filter_known_doc_ids (agentic_rag.py:981):
// it resolves which of a candidate document id set exist within the given
// datasets.
//
// Like Python, it reads the document table directly; the harness package takes
// it through the harness.DocIDVerifier interface so the retrieval path stays
// free of a database dependency.
type DocIDLookup struct {
	docs *dao.DocumentDAO
}

// NewDocIDLookup returns the document-ownership check backed by the document
// table.
func NewDocIDLookup() *DocIDLookup {
	return &DocIDLookup{docs: &dao.DocumentDAO{}}
}

// KnownDocIDs implements harness.DocIDVerifier: it returns the subset of
// candidates that belong to datasetIDs.
func (l *DocIDLookup) KnownDocIDs(ctx context.Context, datasetIDs, candidates []string) (map[string]bool, error) {
	if l == nil || l.docs == nil || len(candidates) == 0 || len(datasetIDs) == 0 {
		return nil, nil
	}
	docs, err := l.docs.GetByIDs(ctx, dao.DB, candidates)
	if err != nil {
		return nil, err
	}
	allowed := make(map[string]bool, len(datasetIDs))
	for _, id := range datasetIDs {
		allowed[id] = true
	}
	known := make(map[string]bool, len(docs))
	for _, d := range docs {
		if d != nil && allowed[d.KbID] {
			known[d.ID] = true
		}
	}
	return known, nil
}

// compile-time check: the lookup satisfies the harness seam.
var _ harness.DocIDVerifier = (*DocIDLookup)(nil)
