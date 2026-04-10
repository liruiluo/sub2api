package apicompat

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestResponsesToChatCompletionsRequest_BasicRoundTrip(t *testing.T) {
	input, err := json.Marshal([]ResponsesInputItem{
		{Role: "system", Content: mustJSONRaw(t, "You are terse.")},
		{Role: "user", Content: mustJSONRaw(t, []ResponsesContentPart{{Type: "input_text", Text: "Say OK"}})},
		{Type: "function_call", CallID: "call_1", Name: "lookup", Arguments: `{"city":"NYC"}`},
		{Type: "function_call_output", CallID: "call_1", Output: "sunny"},
	})
	require.NoError(t, err)

	maxOut := 256
	req := &ResponsesRequest{
		Model:           "gpt-5.4",
		Input:           input,
		MaxOutputTokens: &maxOut,
		Stream:          true,
		Tools: []ResponsesTool{{
			Type:        "function",
			Name:        "lookup",
			Description: "city weather",
			Parameters:  mustJSONRaw(t, map[string]any{"type": "object"}),
		}},
		Reasoning:   &ResponsesReasoning{Effort: "medium"},
		ServiceTier: "flex",
	}

	chatReq, err := ResponsesToChatCompletionsRequest(req)
	require.NoError(t, err)
	require.Equal(t, "gpt-5.4", chatReq.Model)
	require.True(t, chatReq.Stream)
	require.NotNil(t, chatReq.StreamOptions)
	require.True(t, chatReq.StreamOptions.IncludeUsage)
	require.NotNil(t, chatReq.MaxCompletionTokens)
	require.Equal(t, 256, *chatReq.MaxCompletionTokens)
	require.Equal(t, "medium", chatReq.ReasoningEffort)
	require.Equal(t, "flex", chatReq.ServiceTier)
	require.Len(t, chatReq.Messages, 4)
	require.Equal(t, "system", chatReq.Messages[0].Role)
	require.JSONEq(t, `"You are terse."`, string(chatReq.Messages[0].Content))
	require.Equal(t, "user", chatReq.Messages[1].Role)
	require.JSONEq(t, `"Say OK"`, string(chatReq.Messages[1].Content))
	require.Equal(t, "assistant", chatReq.Messages[2].Role)
	require.Len(t, chatReq.Messages[2].ToolCalls, 1)
	require.Equal(t, `{"city":"NYC"}`, chatReq.Messages[2].ToolCalls[0].Function.Arguments)
	require.Equal(t, "tool", chatReq.Messages[3].Role)
	require.JSONEq(t, `"sunny"`, string(chatReq.Messages[3].Content))
	require.Len(t, chatReq.Tools, 1)
	require.Equal(t, "lookup", chatReq.Tools[0].Function.Name)
}

func TestResponsesToChatCompletionsRequest_RejectsUnsupportedTool(t *testing.T) {
	req := &ResponsesRequest{
		Model: "gpt-5.4",
		Input: mustJSONRaw(t, "hi"),
		Tools: []ResponsesTool{{Type: "web_search"}},
	}

	_, err := ResponsesToChatCompletionsRequest(req)
	require.Error(t, err)
	require.Contains(t, err.Error(), "unsupported responses tool")
}

func TestResponsesToChatCompletionsRequest_FillsMissingMessageContentWithEmptyString(t *testing.T) {
	input, err := json.Marshal([]ResponsesInputItem{
		{Role: "system"},
		{Role: "user", Content: mustJSONRaw(t, []ResponsesContentPart{{Type: "input_text", Text: "hi"}})},
	})
	require.NoError(t, err)

	req := &ResponsesRequest{
		Model: "gpt-5.4",
		Input: input,
	}

	chatReq, err := ResponsesToChatCompletionsRequest(req)
	require.NoError(t, err)
	require.Len(t, chatReq.Messages, 2)
	require.JSONEq(t, `""`, string(chatReq.Messages[0].Content))
	require.JSONEq(t, `"hi"`, string(chatReq.Messages[1].Content))
}

func TestResponsesToChatCompletionsRequest_FillsNullMessageContentWithEmptyString(t *testing.T) {
	input := json.RawMessage(`[{"role":"system","content":null},{"role":"user","content":"hi"}]`)

	req := &ResponsesRequest{
		Model: "gpt-5.4",
		Input: input,
	}

	chatReq, err := ResponsesToChatCompletionsRequest(req)
	require.NoError(t, err)
	require.Len(t, chatReq.Messages, 2)
	require.JSONEq(t, `""`, string(chatReq.Messages[0].Content))
	require.JSONEq(t, `"hi"`, string(chatReq.Messages[1].Content))
}

func TestChatCompletionsToResponsesResponse_Basic(t *testing.T) {
	content := mustJSONRaw(t, "done")
	resp := &ChatCompletionsResponse{
		ID:    "chatcmpl_test",
		Model: "mapped-model",
		Choices: []ChatChoice{{
			Index: 0,
			Message: ChatMessage{
				Role:    "assistant",
				Content: content,
				ToolCalls: []ChatToolCall{{
					ID:   "call_1",
					Type: "function",
					Function: ChatFunctionCall{
						Name:      "lookup",
						Arguments: `{"city":"NYC"}`,
					},
				}},
			},
			FinishReason: "tool_calls",
		}},
		Usage: &ChatUsage{PromptTokens: 12, CompletionTokens: 7, TotalTokens: 19, PromptTokensDetails: &ChatTokenDetails{CachedTokens: 3}},
	}

	responsesResp := ChatCompletionsToResponsesResponse(resp, "client-model")
	require.Equal(t, "chatcmpl_test", responsesResp.ID)
	require.Equal(t, "response", responsesResp.Object)
	require.Equal(t, "client-model", responsesResp.Model)
	require.Equal(t, "completed", responsesResp.Status)
	require.Len(t, responsesResp.Output, 2)
	require.Equal(t, "message", responsesResp.Output[0].Type)
	require.Equal(t, "assistant", responsesResp.Output[0].Role)
	require.Equal(t, "done", responsesResp.Output[0].Content[0].Text)
	require.Equal(t, "function_call", responsesResp.Output[1].Type)
	require.Equal(t, "lookup", responsesResp.Output[1].Name)
	require.Equal(t, 12, responsesResp.Usage.InputTokens)
	require.Equal(t, 7, responsesResp.Usage.OutputTokens)
	require.Equal(t, 19, responsesResp.Usage.TotalTokens)
	require.NotNil(t, responsesResp.Usage.InputTokensDetails)
	require.Equal(t, 3, responsesResp.Usage.InputTokensDetails.CachedTokens)
}

func TestChatCompletionsToResponsesResponse_LengthFinishReason(t *testing.T) {
	resp := &ChatCompletionsResponse{
		Choices: []ChatChoice{{
			Message:      ChatMessage{Role: "assistant", Content: mustJSONRaw(t, "partial")},
			FinishReason: "length",
		}},
	}

	responsesResp := ChatCompletionsToResponsesResponse(resp, "gpt-5.4")
	require.Equal(t, "incomplete", responsesResp.Status)
	require.NotNil(t, responsesResp.IncompleteDetails)
	require.Equal(t, "max_output_tokens", responsesResp.IncompleteDetails.Reason)
}

func mustJSONRaw(t *testing.T, v any) json.RawMessage {
	t.Helper()
	data, err := json.Marshal(v)
	require.NoError(t, err)
	return data
}
