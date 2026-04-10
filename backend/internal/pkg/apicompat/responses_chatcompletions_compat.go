package apicompat

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"
)

// ResponsesToChatCompletionsRequest converts a Responses request into a Chat
// Completions request for upstreams that only implement /v1/chat/completions.
func ResponsesToChatCompletionsRequest(req *ResponsesRequest) (*ChatCompletionsRequest, error) {
	if req == nil {
		return nil, fmt.Errorf("responses request is nil")
	}

	messages, err := responsesInputToChatMessages(req.Input)
	if err != nil {
		return nil, err
	}

	out := &ChatCompletionsRequest{
		Model:           req.Model,
		Messages:        messages,
		Temperature:     req.Temperature,
		TopP:            req.TopP,
		Stream:          req.Stream,
		ToolChoice:      req.ToolChoice,
		ServiceTier:     req.ServiceTier,
		ReasoningEffort: "",
	}
	if req.Stream {
		out.StreamOptions = &ChatStreamOptions{IncludeUsage: true}
	}
	if req.MaxOutputTokens != nil {
		v := *req.MaxOutputTokens
		out.MaxCompletionTokens = &v
	}
	if req.Reasoning != nil {
		out.ReasoningEffort = req.Reasoning.Effort
	}
	if len(req.Tools) > 0 {
		tools, err := responsesToolsToChatTools(req.Tools)
		if err != nil {
			return nil, err
		}
		out.Tools = tools
	}

	return out, nil
}

func responsesInputToChatMessages(input json.RawMessage) ([]ChatMessage, error) {
	trimmed := strings.TrimSpace(string(input))
	if trimmed == "" {
		return nil, fmt.Errorf("responses input is required")
	}

	var asString string
	if err := json.Unmarshal(input, &asString); err == nil {
		content, _ := json.Marshal(asString)
		return []ChatMessage{{Role: "user", Content: content}}, nil
	}

	var items []ResponsesInputItem
	if err := json.Unmarshal(input, &items); err != nil {
		return nil, fmt.Errorf("parse responses input: %w", err)
	}

	messages := make([]ChatMessage, 0, len(items))
	for _, item := range items {
		switch item.Type {
		case "function_call":
			args := item.Arguments
			if strings.TrimSpace(args) == "" {
				args = "{}"
			}
			messages = append(messages, ChatMessage{
				Role: "assistant",
				ToolCalls: []ChatToolCall{{
					ID:   item.CallID,
					Type: "function",
					Function: ChatFunctionCall{
						Name:      item.Name,
						Arguments: args,
					},
				}},
			})
		case "function_call_output":
			content, _ := json.Marshal(item.Output)
			messages = append(messages, ChatMessage{
				Role:       "tool",
				ToolCallID: item.CallID,
				Content:    content,
			})
		case "":
			fallthrough
		default:
			role := strings.TrimSpace(item.Role)
			if role == "" {
				role = "user"
			}
			content, err := responsesContentToChatMessageContent(role, item.Content)
			if err != nil {
				return nil, err
			}
			messages = append(messages, ChatMessage{
				Role:    role,
				Content: content,
			})
		}
	}

	return messages, nil
}

func responsesContentToChatMessageContent(role string, content json.RawMessage) (json.RawMessage, error) {
	trimmed := strings.TrimSpace(string(content))
	if trimmed == "" || trimmed == "null" {
		encoded, _ := json.Marshal("")
		return encoded, nil
	}

	var asString string
	if err := json.Unmarshal(content, &asString); err == nil {
		encoded, _ := json.Marshal(asString)
		return encoded, nil
	}

	var parts []ResponsesContentPart
	if err := json.Unmarshal(content, &parts); err != nil {
		return nil, fmt.Errorf("parse responses %s content: %w", role, err)
	}
	if len(parts) == 0 {
		encoded, _ := json.Marshal("")
		return encoded, nil
	}

	chatParts := make([]ChatContentPart, 0, len(parts))
	var textBuilder strings.Builder
	onlyText := true
	for _, part := range parts {
		switch part.Type {
		case "input_text", "output_text", "text":
			if part.Text == "" {
				continue
			}
			if onlyText {
				textBuilder.WriteString(part.Text)
			}
			chatParts = append(chatParts, ChatContentPart{Type: "text", Text: part.Text})
		case "input_image":
			onlyText = false
			if part.ImageURL == "" {
				continue
			}
			chatParts = append(chatParts, ChatContentPart{
				Type:     "image_url",
				ImageURL: &ChatImageURL{URL: part.ImageURL},
			})
		default:
			return nil, fmt.Errorf("unsupported responses content part for chat compat: %s", part.Type)
		}
	}

	if onlyText {
		encoded, _ := json.Marshal(textBuilder.String())
		return encoded, nil
	}
	encoded, err := json.Marshal(chatParts)
	if err != nil {
		return nil, err
	}
	return encoded, nil
}

func responsesToolsToChatTools(tools []ResponsesTool) ([]ChatTool, error) {
	out := make([]ChatTool, 0, len(tools))
	for _, tool := range tools {
		toolType := strings.TrimSpace(tool.Type)
		if toolType == "" || toolType == "function" {
			out = append(out, ChatTool{
				Type: "function",
				Function: &ChatFunction{
					Name:        tool.Name,
					Description: tool.Description,
					Parameters:  tool.Parameters,
					Strict:      tool.Strict,
				},
			})
			continue
		}
		return nil, fmt.Errorf("unsupported responses tool for chat compat: %s", toolType)
	}
	return out, nil
}

// ChatCompletionsToResponsesResponse converts a Chat Completions response into a
// Responses API response.
func ChatCompletionsToResponsesResponse(resp *ChatCompletionsResponse, model string) *ResponsesResponse {
	id := ""
	if resp != nil {
		id = strings.TrimSpace(resp.ID)
	}
	if id == "" {
		id = generateResponseID()
	}

	resolvedModel := strings.TrimSpace(model)
	if resolvedModel == "" && resp != nil {
		resolvedModel = strings.TrimSpace(resp.Model)
	}

	out := &ResponsesResponse{
		ID:     id,
		Object: "response",
		Model:  resolvedModel,
		Status: "completed",
	}
	if resp == nil || len(resp.Choices) == 0 {
		return out
	}

	choice := resp.Choices[0]
	message := choice.Message
	messageParts := chatMessageContentToResponsesOutputParts(message.Content)
	if len(messageParts) > 0 {
		out.Output = append(out.Output, ResponsesOutput{
			Type:    "message",
			ID:      id + "_msg_0",
			Role:    "assistant",
			Content: messageParts,
			Status:  "completed",
		})
	}
	for idx, toolCall := range message.ToolCalls {
		out.Output = append(out.Output, ResponsesOutput{
			Type:      "function_call",
			CallID:    toolCall.ID,
			Name:      toolCall.Function.Name,
			Arguments: toolCall.Function.Arguments,
			ID:        fmt.Sprintf("%s_fc_%d", id, idx),
		})
	}

	switch choice.FinishReason {
	case "length":
		out.Status = "incomplete"
		out.IncompleteDetails = &ResponsesIncompleteDetails{Reason: "max_output_tokens"}
	case "content_filter":
		out.Status = "incomplete"
		out.IncompleteDetails = &ResponsesIncompleteDetails{Reason: "content_filter"}
	default:
		out.Status = "completed"
	}

	if resp.Usage != nil {
		out.Usage = &ResponsesUsage{
			InputTokens:  resp.Usage.PromptTokens,
			OutputTokens: resp.Usage.CompletionTokens,
			TotalTokens:  resp.Usage.TotalTokens,
		}
		if resp.Usage.PromptTokensDetails != nil && resp.Usage.PromptTokensDetails.CachedTokens > 0 {
			out.Usage.InputTokensDetails = &ResponsesInputTokensDetails{
				CachedTokens: resp.Usage.PromptTokensDetails.CachedTokens,
			}
		}
	}

	return out
}

func chatMessageContentToResponsesOutputParts(content json.RawMessage) []ResponsesContentPart {
	if len(content) == 0 {
		return nil
	}

	var asString string
	if err := json.Unmarshal(content, &asString); err == nil {
		if asString == "" {
			return nil
		}
		return []ResponsesContentPart{{Type: "output_text", Text: asString}}
	}

	var chatParts []ChatContentPart
	if err := json.Unmarshal(content, &chatParts); err != nil {
		return nil
	}
	out := make([]ResponsesContentPart, 0, len(chatParts))
	for _, part := range chatParts {
		switch part.Type {
		case "text":
			if part.Text != "" {
				out = append(out, ResponsesContentPart{Type: "output_text", Text: part.Text})
			}
		case "image_url":
			// Responses output does not expose image_url assistant parts in the same way;
			// ignore them rather than fabricating an unsupported shape.
		}
	}
	return out
}

func generateResponseID() string {
	b := make([]byte, 12)
	_, _ = rand.Read(b)
	return "resp_" + hex.EncodeToString(b)
}
