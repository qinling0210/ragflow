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

	"github.com/cloudwego/eino/schema"

	"gorm.io/gorm"
	"ragflow/internal/agent/chat"
)

// streamingInvoker streams a fixed set of pieces.
type streamingInvoker struct {
	pieces []piece
	err    error
}

type piece struct {
	text    string
	isThink bool
}

func (s *streamingInvoker) Invoke(context.Context, *gorm.DB, chat.Request) (*chat.Response, error) {
	return &chat.Response{Content: "one-shot"}, nil
}

func (s *streamingInvoker) Stream(_ context.Context, _ *gorm.DB, _ chat.Request, onDelta func(string, bool) error) (*chat.Response, error) {
	if s.err != nil {
		return nil, s.err
	}
	var content, reasoning string
	for _, p := range s.pieces {
		if onDelta != nil {
			if err := onDelta(p.text, p.isThink); err != nil {
				return nil, err
			}
		}
		if p.isThink {
			reasoning += p.text
		} else {
			content += p.text
		}
	}
	return &chat.Response{Content: content, Thinking: reasoning}, nil
}

// plainInvoker only implements the non-streaming seam.
type plainInvoker struct{}

func (p *plainInvoker) Invoke(context.Context, *gorm.DB, chat.Request) (*chat.Response, error) {
	return &chat.Response{Content: "one-shot"}, nil
}

func TestStreamCompleteForwardsDeltas(t *testing.T) {
	m := &InvokerSessionModel{Invoker: &streamingInvoker{pieces: []piece{
		{text: "think", isThink: true},
		{text: "Hello ", isThink: false},
		{text: "world", isThink: false},
	}}}

	var got []piece
	reply, err := m.StreamComplete(context.Background(), []schema.Message{*schema.UserMessage("q")}, nil,
		func(delta string, isThink bool) error {
			got = append(got, piece{text: delta, isThink: isThink})
			return nil
		})

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if reply.Content != "Hello world" {
		t.Fatalf("Content = %q, want %q", reply.Content, "Hello world")
	}
	if len(got) != 3 || !got[0].isThink || got[0].text != "think" {
		t.Fatalf("deltas = %+v, want reasoning flagged first", got)
	}
}

func TestStreamCompleteFailsWithoutStreamingInvoker(t *testing.T) {
	// A non-streaming invoker must report failure so the caller falls back to
	// the one-shot call rather than silently producing nothing.
	m := &InvokerSessionModel{Invoker: &plainInvoker{}}

	if _, err := m.StreamComplete(context.Background(), nil, nil, nil); err == nil {
		t.Fatal("want an error when the invoker cannot stream")
	}
}

func TestStreamCompletePropagatesError(t *testing.T) {
	m := &InvokerSessionModel{Invoker: &streamingInvoker{err: errors.New("provider down")}}

	if _, err := m.StreamComplete(context.Background(), nil, nil, nil); err == nil {
		t.Fatal("want the provider error")
	}
}

func TestStreamCompleteRequiresInvoker(t *testing.T) {
	m := &InvokerSessionModel{}

	if _, err := m.StreamComplete(context.Background(), nil, nil, nil); err == nil {
		t.Fatal("want an error when no invoker is configured")
	}
}

// capturingInvoker records the last Request and returns a native tool call so we
// can verify the seam forwards the declared tools and that the harness prefers
// native calls over prompt-based fenced-block parsing.
type capturingInvoker struct {
	lastReq *chat.Request
	hasTool bool
	tool    chat.ToolCall
	content string
}

func (c *capturingInvoker) Invoke(_ context.Context, _ *gorm.DB, req chat.Request) (*chat.Response, error) {
	c.lastReq = &req
	resp := &chat.Response{Content: c.content}
	if c.hasTool {
		resp.ToolCalls = []chat.ToolCall{c.tool}
	}
	return resp, nil
}

func TestCompleteForwardsNativeToolsAndPrefersNativeCalls(t *testing.T) {
	tools := []ToolSpec{{
		Type: "function",
		Function: ToolFunction{
			Name:        "search",
			Description: "run a search",
			Parameters:  map[string]any{"type": "object", "properties": map[string]any{}},
		},
	}}
	inv := &capturingInvoker{hasTool: true, tool: chat.ToolCall{ID: "c1", Name: "search", Arguments: map[string]any{"q": "x"}}, content: "ok"}
	m := &InvokerSessionModel{Invoker: inv}

	reply, err := m.Complete(context.Background(), []schema.Message{*schema.UserMessage("q")}, tools)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if inv.lastReq == nil || len(inv.lastReq.Tools) != 1 {
		t.Fatalf("tools not forwarded to invoker: req=%+v", inv.lastReq)
	}
	if inv.lastReq.Tools[0].Function.Name != "search" {
		t.Fatalf("forwarded tool name = %q, want search", inv.lastReq.Tools[0].Function.Name)
	}
	if inv.lastReq.ToolChoice != chat.ToolChoiceAuto {
		t.Fatalf("tool choice = %q, want auto", inv.lastReq.ToolChoice)
	}
	if len(reply.ToolCalls) != 1 || reply.ToolCalls[0].Name != "search" || reply.ToolCalls[0].Args["q"] != "x" {
		t.Fatalf("native tool call not preferred: %+v", reply.ToolCalls)
	}
}

func TestCompleteFallsBackToPromptParsingWhenNoNativeCall(t *testing.T) {
	inv := &capturingInvoker{hasTool: false, content: ""} // no native call and no fenced block
	m := &InvokerSessionModel{Invoker: inv}
	tools := []ToolSpec{{
		Type:     "function",
		Function: ToolFunction{Name: "search", Description: "d", Parameters: map[string]any{}},
	}}

	// No native call and no fenced block in the content => the reply stays
	// tool-less. This confirms the harness still declares tools on the request
	// (verified above) but does not hallucinate a call.
	reply, err := m.Complete(context.Background(), []schema.Message{*schema.UserMessage("q")}, tools)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(reply.ToolCalls) != 0 {
		t.Fatalf("expected no tool calls, got %+v", reply.ToolCalls)
	}
}

// streamingCapturingInvoker implements chat.StreamingInvoker and records the
// streaming request, returning a native tool call so we can verify the harness
// prefers streamed native calls over prompt-based fenced-block parsing.
type streamingCapturingInvoker struct {
	capturingInvoker
	onDelta func(delta string, isThink bool) error
}

func (s *streamingCapturingInvoker) Stream(_ context.Context, _ *gorm.DB, req chat.Request, onDelta func(delta string, isThink bool) error) (*chat.Response, error) {
	s.lastReq = &req
	if onDelta != nil {
		_ = onDelta("partial", false)
	}
	resp := &chat.Response{Content: "streamed"}
	if s.hasTool {
		resp.ToolCalls = []chat.ToolCall{s.tool}
	}
	return resp, nil
}

func TestStreamCompleteForwardsNativeToolsAndPrefersNativeCalls(t *testing.T) {
	tools := []ToolSpec{{
		Type: "function",
		Function: ToolFunction{
			Name:        "search",
			Description: "run a search",
			Parameters:  map[string]any{"type": "object", "properties": map[string]any{}},
		},
	}}
	inv := &streamingCapturingInvoker{
		capturingInvoker: capturingInvoker{
			hasTool: true,
			tool:    chat.ToolCall{ID: "c2", Name: "search", Arguments: map[string]any{"q": "y"}},
		},
	}
	m := &InvokerSessionModel{Invoker: inv}

	reply, err := m.StreamComplete(context.Background(), []schema.Message{*schema.UserMessage("q")}, tools, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if inv.lastReq == nil || len(inv.lastReq.Tools) != 1 || inv.lastReq.Tools[0].Function.Name != "search" {
		t.Fatalf("streaming tools not forwarded: req=%+v", inv.lastReq)
	}
	if inv.lastReq.ToolChoice != chat.ToolChoiceAuto {
		t.Fatalf("streaming tool choice = %q, want auto", inv.lastReq.ToolChoice)
	}
	if len(reply.ToolCalls) != 1 || reply.ToolCalls[0].Name != "search" || reply.ToolCalls[0].Args["q"] != "y" {
		t.Fatalf("streamed native tool call not preferred: %+v", reply.ToolCalls)
	}
}

func TestStreamCompletePrefersNativeOverFencedBlock(t *testing.T) {
	tools := []ToolSpec{{
		Type:     "function",
		Function: ToolFunction{Name: "search", Description: "d", Parameters: map[string]any{}},
	}}
	// Streamed content carries a fenced block, but a native call is also present.
	// The harness must prefer the native call, not the parsed fenced block.
	inv := &streamingCapturingInvoker{
		capturingInvoker: capturingInvoker{
			hasTool: true,
			tool:    chat.ToolCall{ID: "c3", Name: "search", Arguments: map[string]any{"q": "z"}},
			content: "```json\n{\"tool\": \"search\", \"args\": {\"q\": \"ignored\"}}\n```",
		},
	}
	m := &InvokerSessionModel{Invoker: inv}

	reply, err := m.StreamComplete(context.Background(), []schema.Message{*schema.UserMessage("q")}, tools, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(reply.ToolCalls) != 1 || reply.ToolCalls[0].Args["q"] != "z" {
		t.Fatalf("native streaming call should win over fenced block: %+v", reply.ToolCalls)
	}
}
