package service

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/pkg/apicompat"
	"github.com/Wei-Shaw/sub2api/internal/pkg/logger"
	"github.com/Wei-Shaw/sub2api/internal/util/responseheaders"
	"github.com/gin-gonic/gin"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
	"go.uber.org/zap"
)

const openAIAPIKeyChatCompletionsCompatExtraKey = "openai_apikey_chat_completions_compat"

func isOpenAIChatCompletionsCompatError(statusCode int, body []byte) bool {
	combined := strings.ToLower(strings.TrimSpace(extractUpstreamErrorMessage(body)))
	if combined == "" {
		combined = strings.ToLower(strings.TrimSpace(string(body)))
	} else {
		combined += " " + strings.ToLower(strings.TrimSpace(string(body)))
	}
	if statusCode == http.StatusInternalServerError &&
		strings.Contains(combined, "not implemented") &&
		(strings.Contains(combined, "convert_request_failed") || strings.Contains(combined, "new_api_error")) {
		return true
	}
	if statusCode != http.StatusBadRequest && statusCode != http.StatusNotFound {
		return false
	}
	return strings.Contains(combined, "field messages is required") ||
		strings.Contains(combined, `"messages" is required`) ||
		(statusCode == http.StatusNotFound && strings.Contains(combined, "page not found"))
}

func buildOpenAIChatCompletionsURL(base string) string {
	normalized := strings.TrimRight(strings.TrimSpace(base), "/")
	if strings.HasSuffix(normalized, "/chat/completions") {
		return normalized
	}
	if strings.HasSuffix(normalized, "/v1") {
		return normalized + "/chat/completions"
	}
	return normalized + "/v1/chat/completions"
}

func (s *OpenAIGatewayService) markOpenAIChatCompletionsCompat(ctx context.Context, account *Account) {
	if account == nil || !account.IsOpenAIApiKey() || account.IsOpenAIChatCompletionsCompatEnabled() {
		return
	}
	updates := map[string]any{openAIAPIKeyChatCompletionsCompatExtraKey: true}
	mergeAccountExtra(account, updates)
	if s == nil || s.accountRepo == nil {
		return
	}
	if err := s.accountRepo.UpdateExtra(ctx, account.ID, updates); err != nil {
		logger.FromContext(ctx).With(
			zap.Int64("account_id", account.ID),
			zap.String("account_name", account.Name),
			zap.Error(err),
		).Warn("openai chat compat: failed to persist compat flag")
	}
}

func (s *OpenAIGatewayService) buildUpstreamChatCompletionsRequest(
	ctx context.Context,
	c *gin.Context,
	account *Account,
	body []byte,
	token string,
	isStream bool,
) (*http.Request, error) {
	baseURL := account.GetOpenAIBaseURL()
	if baseURL == "" {
		baseURL = "https://api.openai.com"
	}
	validatedURL, err := s.validateUpstreamBaseURL(baseURL)
	if err != nil {
		return nil, err
	}
	targetURL := buildOpenAIChatCompletionsURL(validatedURL)

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, targetURL, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}

	for key, values := range c.Request.Header {
		lowerKey := strings.ToLower(key)
		if lowerKey != "accept" && lowerKey != "openai-beta" && !openaiAllowedHeaders[lowerKey] {
			continue
		}
		for _, v := range values {
			req.Header.Add(key, v)
		}
	}

	req.Header.Del("authorization")
	req.Header.Set("authorization", "Bearer "+token)
	if req.Header.Get("content-type") == "" {
		req.Header.Set("content-type", "application/json")
	}
	if req.Header.Get("accept") == "" {
		if isStream {
			req.Header.Set("accept", "text/event-stream")
		} else {
			req.Header.Set("accept", "application/json")
		}
	}
	if customUA := account.GetOpenAIUserAgent(); customUA != "" {
		req.Header.Set("user-agent", customUA)
	}
	if s.cfg != nil && s.cfg.Gateway.ForceCodexCLI {
		req.Header.Set("user-agent", codexCLIUserAgent)
	}

	return req, nil
}

func (s *OpenAIGatewayService) forwardOpenAIChatCompletionsCompat(
	ctx context.Context,
	c *gin.Context,
	account *Account,
	chatBody []byte,
	startTime time.Time,
	originalModel string,
	mappedModel string,
	clientStream bool,
) (*OpenAIForwardResult, error) {
	token, _, err := s.GetAccessToken(ctx, account)
	if err != nil {
		return nil, err
	}
	upstreamReq, err := s.buildUpstreamChatCompletionsRequest(ctx, c, account, chatBody, token, clientStream)
	if err != nil {
		return nil, err
	}

	proxyURL := ""
	if account.ProxyID != nil && account.Proxy != nil {
		proxyURL = account.Proxy.URL()
	}

	upstreamStart := time.Now()
	resp, err := s.httpUpstream.Do(upstreamReq, proxyURL, account.ID, account.Concurrency)
	SetOpsLatencyMs(c, OpsUpstreamLatencyMsKey, time.Since(upstreamStart).Milliseconds())
	if err != nil {
		safeErr := sanitizeUpstreamErrorMessage(err.Error())
		setOpsUpstreamError(c, 0, safeErr, "")
		appendOpsUpstreamError(c, OpsUpstreamErrorEvent{
			Platform:           account.Platform,
			AccountID:          account.ID,
			AccountName:        account.Name,
			UpstreamStatusCode: 0,
			Kind:               "request_error",
			Message:            safeErr,
		})
		if shouldFailoverOpenAIRequestError(err) {
			return nil, &UpstreamFailoverError{StatusCode: http.StatusBadGateway}
		}
		writeChatCompletionsError(c, http.StatusBadGateway, "upstream_error", "Upstream request failed")
		return nil, fmt.Errorf("upstream request failed: %s", safeErr)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode >= 400 {
		respBody, _ := io.ReadAll(io.LimitReader(resp.Body, 2<<20))
		_ = resp.Body.Close()
		resp.Body = io.NopCloser(bytes.NewReader(respBody))

		upstreamMsg := strings.TrimSpace(extractUpstreamErrorMessage(respBody))
		upstreamMsg = sanitizeUpstreamErrorMessage(upstreamMsg)
		if s.shouldFailoverOpenAIUpstreamResponse(resp.StatusCode, upstreamMsg, respBody) {
			appendOpsUpstreamError(c, OpsUpstreamErrorEvent{
				Platform:           account.Platform,
				AccountID:          account.ID,
				AccountName:        account.Name,
				UpstreamStatusCode: resp.StatusCode,
				UpstreamRequestID:  resp.Header.Get("x-request-id"),
				Kind:               "failover",
				Message:            upstreamMsg,
			})
			if s.rateLimitService != nil {
				s.rateLimitService.HandleUpstreamError(ctx, account, resp.StatusCode, resp.Header, respBody)
			}
			return nil, &UpstreamFailoverError{StatusCode: resp.StatusCode, ResponseBody: respBody}
		}
		resp.Body = io.NopCloser(bytes.NewReader(respBody))
		return s.handleChatCompletionsErrorResponse(resp, c, account)
	}

	var usage *OpenAIUsage
	var firstTokenMs *int
	if clientStream {
		streamResult, err := s.handleChatCompletionsStreamingCompatResponse(resp, c, account, startTime, originalModel, mappedModel)
		if err != nil {
			return nil, err
		}
		usage = streamResult.usage
		firstTokenMs = streamResult.firstTokenMs
	} else {
		usage, err = s.handleChatCompletionsNonStreamingCompatResponse(resp, c, originalModel, mappedModel)
		if err != nil {
			return nil, err
		}
	}
	if usage == nil {
		usage = &OpenAIUsage{}
	}

	return &OpenAIForwardResult{
		RequestID:       resp.Header.Get("x-request-id"),
		Usage:           *usage,
		Model:           originalModel,
		ServiceTier:     extractOpenAIServiceTierFromBody(chatBody),
		ReasoningEffort: extractOpenAIReasoningEffortFromBody(chatBody, originalModel),
		Stream:          clientStream,
		Duration:        time.Since(startTime),
		FirstTokenMs:    firstTokenMs,
	}, nil
}

func (s *OpenAIGatewayService) handleChatCompletionsStreamingCompatResponse(
	resp *http.Response,
	c *gin.Context,
	account *Account,
	startTime time.Time,
	originalModel string,
	mappedModel string,
) (*openaiStreamingResult, error) {
	if s.responseHeaderFilter != nil {
		responseheaders.WriteFilteredHeaders(c.Writer.Header(), resp.Header, s.responseHeaderFilter)
	}
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Header("X-Accel-Buffering", "no")
	if v := resp.Header.Get("x-request-id"); v != "" {
		c.Header("x-request-id", v)
	}

	w := c.Writer
	flusher, ok := w.(http.Flusher)
	if !ok {
		return nil, errors.New("streaming not supported")
	}

	usage := &OpenAIUsage{}
	var firstTokenMs *int
	clientDisconnected := false
	needModelReplace := originalModel != mappedModel

	scanner := bufio.NewScanner(resp.Body)
	maxLineSize := defaultMaxLineSize
	if s.cfg != nil && s.cfg.Gateway.MaxLineSize > 0 {
		maxLineSize = s.cfg.Gateway.MaxLineSize
	}
	scanBuf := getSSEScannerBuf64K()
	scanner.Buffer(scanBuf[:0], maxLineSize)
	defer putSSEScannerBuf64K(scanBuf)

	for scanner.Scan() {
		line := scanner.Text()
		if data, ok := extractOpenAISSEDataLine(line); ok {
			trimmed := strings.TrimSpace(data)
			if needModelReplace && trimmed != "" && trimmed != "[DONE]" && strings.Contains(trimmed, mappedModel) {
				line = replaceChatCompletionsModelInSSELine(line, mappedModel, originalModel)
				data, _ = extractOpenAISSEDataLine(line)
			}
			if firstTokenMs == nil && trimmed != "" && trimmed != "[DONE]" {
				ms := int(time.Since(startTime).Milliseconds())
				firstTokenMs = &ms
			}
			parseChatCompletionsSSEUsageBytes([]byte(data), usage)
		}
		if !clientDisconnected {
			if _, err := fmt.Fprintln(w, line); err != nil {
				clientDisconnected = true
				logger.LegacyPrintf("service.openai_gateway", "[OpenAI chat compat] client disconnected during streaming, draining upstream: account=%d", account.ID)
			} else {
				flusher.Flush()
			}
		}
	}
	if err := scanner.Err(); err != nil {
		if clientDisconnected || errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return &openaiStreamingResult{usage: usage, firstTokenMs: firstTokenMs}, nil
		}
		if errors.Is(err, bufio.ErrTooLong) {
			return &openaiStreamingResult{usage: usage, firstTokenMs: firstTokenMs}, err
		}
		if shouldFailoverOpenAIStreamingFailure(c, firstTokenMs, clientDisconnected) {
			return &openaiStreamingResult{usage: usage, firstTokenMs: firstTokenMs}, &UpstreamFailoverError{StatusCode: http.StatusBadGateway}
		}
		return &openaiStreamingResult{usage: usage, firstTokenMs: firstTokenMs}, fmt.Errorf("stream read error: %w", err)
	}
	return &openaiStreamingResult{usage: usage, firstTokenMs: firstTokenMs}, nil
}

func (s *OpenAIGatewayService) handleChatCompletionsNonStreamingCompatResponse(
	resp *http.Response,
	c *gin.Context,
	originalModel string,
	mappedModel string,
) (*OpenAIUsage, error) {
	maxBytes := resolveUpstreamResponseReadLimit(s.cfg)
	body, err := readUpstreamResponseBodyLimited(resp.Body, maxBytes)
	if err != nil {
		return nil, err
	}
	usageValue, usageOK := extractChatCompletionsUsageFromJSONBytes(body)
	if !usageOK {
		return nil, &UpstreamFailoverError{StatusCode: http.StatusBadGateway}
	}
	if originalModel != mappedModel {
		body = replaceChatCompletionsModelInResponseBody(body, mappedModel, originalModel)
	}
	responseheaders.WriteFilteredHeaders(c.Writer.Header(), resp.Header, s.responseHeaderFilter)
	contentType := resp.Header.Get("Content-Type")
	if contentType == "" {
		contentType = "application/json"
	}
	c.Data(resp.StatusCode, contentType, body)
	return &usageValue, nil
}

func (s *OpenAIGatewayService) forwardOpenAIResponsesViaChatCompletionsCompat(
	ctx context.Context,
	c *gin.Context,
	account *Account,
	body []byte,
	startTime time.Time,
	originalModel string,
	mappedModel string,
	reqStream bool,
) (*OpenAIForwardResult, error) {
	var responsesReq apicompat.ResponsesRequest
	if err := json.Unmarshal(body, &responsesReq); err != nil {
		return nil, fmt.Errorf("parse responses request for chat compat: %w", err)
	}
	chatReq, err := apicompat.ResponsesToChatCompletionsRequest(&responsesReq)
	if err != nil {
		setOpsUpstreamError(c, http.StatusBadGateway, err.Error(), "")
		return nil, &UpstreamFailoverError{StatusCode: http.StatusBadGateway}
	}
	chatReq.Stream = false
	chatReq.StreamOptions = nil
	chatBody, err := json.Marshal(chatReq)
	if err != nil {
		return nil, fmt.Errorf("marshal chat compat request: %w", err)
	}

	token, _, err := s.GetAccessToken(ctx, account)
	if err != nil {
		return nil, err
	}
	upstreamReq, err := s.buildUpstreamChatCompletionsRequest(ctx, c, account, chatBody, token, false)
	if err != nil {
		return nil, err
	}

	proxyURL := ""
	if account.ProxyID != nil && account.Proxy != nil {
		proxyURL = account.Proxy.URL()
	}

	upstreamStart := time.Now()
	resp, err := s.httpUpstream.Do(upstreamReq, proxyURL, account.ID, account.Concurrency)
	SetOpsLatencyMs(c, OpsUpstreamLatencyMsKey, time.Since(upstreamStart).Milliseconds())
	if err != nil {
		safeErr := sanitizeUpstreamErrorMessage(err.Error())
		setOpsUpstreamError(c, 0, safeErr, "")
		appendOpsUpstreamError(c, OpsUpstreamErrorEvent{
			Platform:           account.Platform,
			AccountID:          account.ID,
			AccountName:        account.Name,
			UpstreamStatusCode: 0,
			Kind:               "request_error",
			Message:            safeErr,
		})
		if shouldFailoverOpenAIRequestError(err) {
			return nil, &UpstreamFailoverError{StatusCode: http.StatusBadGateway}
		}
		c.JSON(http.StatusBadGateway, gin.H{"error": gin.H{"type": "upstream_error", "message": "Upstream request failed"}})
		return nil, fmt.Errorf("upstream request failed: %s", safeErr)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode >= 400 {
		respBody, _ := io.ReadAll(io.LimitReader(resp.Body, 2<<20))
		_ = resp.Body.Close()
		resp.Body = io.NopCloser(bytes.NewReader(respBody))

		upstreamMsg := strings.TrimSpace(extractUpstreamErrorMessage(respBody))
		upstreamMsg = sanitizeUpstreamErrorMessage(upstreamMsg)
		if s.shouldFailoverOpenAIUpstreamResponse(resp.StatusCode, upstreamMsg, respBody) {
			appendOpsUpstreamError(c, OpsUpstreamErrorEvent{
				Platform:           account.Platform,
				AccountID:          account.ID,
				AccountName:        account.Name,
				UpstreamStatusCode: resp.StatusCode,
				UpstreamRequestID:  resp.Header.Get("x-request-id"),
				Kind:               "failover",
				Message:            upstreamMsg,
			})
			s.handleFailoverSideEffects(ctx, resp, account)
			return nil, &UpstreamFailoverError{StatusCode: resp.StatusCode, ResponseBody: respBody}
		}
		resp.Body = io.NopCloser(bytes.NewReader(respBody))
		return s.handleErrorResponse(ctx, resp, c, account, body)
	}

	usage, firstTokenMs, err := s.handleResponsesViaChatCompletionsCompatSuccess(resp, c, originalModel, reqStream, startTime)
	if err != nil {
		return nil, err
	}
	if usage == nil {
		usage = &OpenAIUsage{}
	}
	return &OpenAIForwardResult{
		RequestID:       resp.Header.Get("x-request-id"),
		Usage:           *usage,
		Model:           originalModel,
		ServiceTier:     extractOpenAIServiceTierFromBody(body),
		ReasoningEffort: extractOpenAIReasoningEffortFromBody(body, originalModel),
		Stream:          reqStream,
		Duration:        time.Since(startTime),
		FirstTokenMs:    firstTokenMs,
	}, nil
}

func (s *OpenAIGatewayService) handleResponsesViaChatCompletionsCompatSuccess(
	resp *http.Response,
	c *gin.Context,
	originalModel string,
	reqStream bool,
	startTime time.Time,
) (*OpenAIUsage, *int, error) {
	maxBytes := resolveUpstreamResponseReadLimit(s.cfg)
	body, err := readUpstreamResponseBodyLimited(resp.Body, maxBytes)
	if err != nil {
		return nil, nil, err
	}
	usageValue, usageOK := extractChatCompletionsUsageFromJSONBytes(body)
	if !usageOK {
		return nil, nil, &UpstreamFailoverError{StatusCode: http.StatusBadGateway}
	}
	var chatResp apicompat.ChatCompletionsResponse
	if err := json.Unmarshal(body, &chatResp); err != nil {
		return nil, nil, &UpstreamFailoverError{StatusCode: http.StatusBadGateway}
	}
	responsesResp := apicompat.ChatCompletionsToResponsesResponse(&chatResp, originalModel)
	if reqStream {
		firstTokenMs, err := s.writeResponsesCompatStream(c, resp, responsesResp, startTime)
		return &usageValue, firstTokenMs, err
	}

	encoded, err := json.Marshal(responsesResp)
	if err != nil {
		return nil, nil, fmt.Errorf("marshal responses compat response: %w", err)
	}
	responseheaders.WriteFilteredHeaders(c.Writer.Header(), resp.Header, s.responseHeaderFilter)
	c.Writer.Header().Set("Content-Type", "application/json")
	if v := resp.Header.Get("x-request-id"); v != "" {
		c.Writer.Header().Set("x-request-id", v)
	}
	c.Data(resp.StatusCode, "application/json", encoded)
	return &usageValue, nil, nil
}

func (s *OpenAIGatewayService) writeResponsesCompatStream(
	c *gin.Context,
	upstreamResp *http.Response,
	responsesResp *apicompat.ResponsesResponse,
	startTime time.Time,
) (*int, error) {
	if s.responseHeaderFilter != nil {
		responseheaders.WriteFilteredHeaders(c.Writer.Header(), upstreamResp.Header, s.responseHeaderFilter)
	}
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Header("X-Accel-Buffering", "no")
	if v := upstreamResp.Header.Get("x-request-id"); v != "" {
		c.Header("x-request-id", v)
	}
	w := c.Writer
	flusher, ok := w.(http.Flusher)
	if !ok {
		return nil, errors.New("streaming not supported")
	}

	sequence := 1
	writeEvent := func(evt apicompat.ResponsesStreamEvent) error {
		if evt.SequenceNumber == 0 {
			evt.SequenceNumber = sequence
			sequence++
		}
		data, err := json.Marshal(evt)
		if err != nil {
			return err
		}
		if _, err := fmt.Fprintf(w, "data: %s\n\n", data); err != nil {
			return err
		}
		flusher.Flush()
		return nil
	}

	created := &apicompat.ResponsesResponse{ID: responsesResp.ID, Object: "response", Model: responsesResp.Model, Status: "in_progress"}
	if err := writeEvent(apicompat.ResponsesStreamEvent{Type: "response.created", Response: created}); err != nil {
		return nil, err
	}

	var firstTokenMs *int
	for idx, item := range responsesResp.Output {
		switch item.Type {
		case "message":
			added := item
			added.Status = "in_progress"
			added.Content = nil
			if err := writeEvent(apicompat.ResponsesStreamEvent{Type: "response.output_item.added", OutputIndex: idx, Item: &added}); err != nil {
				return firstTokenMs, err
			}
			for contentIdx, part := range item.Content {
				if part.Type != "output_text" || part.Text == "" {
					continue
				}
				if firstTokenMs == nil {
					ms := int(time.Since(startTime).Milliseconds())
					firstTokenMs = &ms
				}
				if err := writeEvent(apicompat.ResponsesStreamEvent{Type: "response.output_text.delta", OutputIndex: idx, ContentIndex: contentIdx, Delta: part.Text, ItemID: item.ID}); err != nil {
					return firstTokenMs, err
				}
				if err := writeEvent(apicompat.ResponsesStreamEvent{Type: "response.output_text.done", OutputIndex: idx, ContentIndex: contentIdx, Text: part.Text, ItemID: item.ID}); err != nil {
					return firstTokenMs, err
				}
			}
			if err := writeEvent(apicompat.ResponsesStreamEvent{Type: "response.output_item.done", OutputIndex: idx, Item: &item}); err != nil {
				return firstTokenMs, err
			}
		case "function_call":
			added := item
			added.Arguments = ""
			if err := writeEvent(apicompat.ResponsesStreamEvent{Type: "response.output_item.added", OutputIndex: idx, Item: &added}); err != nil {
				return firstTokenMs, err
			}
			if strings.TrimSpace(item.Arguments) != "" {
				if firstTokenMs == nil {
					ms := int(time.Since(startTime).Milliseconds())
					firstTokenMs = &ms
				}
				if err := writeEvent(apicompat.ResponsesStreamEvent{Type: "response.function_call_arguments.delta", OutputIndex: idx, CallID: item.CallID, Name: item.Name, Delta: item.Arguments}); err != nil {
					return firstTokenMs, err
				}
				if err := writeEvent(apicompat.ResponsesStreamEvent{Type: "response.function_call_arguments.done", OutputIndex: idx, CallID: item.CallID, Name: item.Name, Arguments: item.Arguments}); err != nil {
					return firstTokenMs, err
				}
			}
			if err := writeEvent(apicompat.ResponsesStreamEvent{Type: "response.output_item.done", OutputIndex: idx, Item: &item}); err != nil {
				return firstTokenMs, err
			}
		}
	}

	if err := writeEvent(apicompat.ResponsesStreamEvent{Type: "response.completed", Response: responsesResp}); err != nil {
		return firstTokenMs, err
	}
	if _, err := fmt.Fprint(w, "data: [DONE]\n\n"); err != nil {
		return firstTokenMs, err
	}
	flusher.Flush()
	return firstTokenMs, nil
}

func extractChatCompletionsUsageFromJSONBytes(body []byte) (OpenAIUsage, bool) {
	if len(body) == 0 || !gjson.ValidBytes(body) {
		return OpenAIUsage{}, false
	}
	if !gjson.GetBytes(body, "usage.total_tokens").Exists() {
		return OpenAIUsage{}, false
	}
	return OpenAIUsage{
		InputTokens:          int(gjson.GetBytes(body, "usage.prompt_tokens").Int()),
		OutputTokens:         int(gjson.GetBytes(body, "usage.completion_tokens").Int()),
		CacheReadInputTokens: int(gjson.GetBytes(body, "usage.prompt_tokens_details.cached_tokens").Int()),
	}, true
}

func parseChatCompletionsSSEUsageBytes(data []byte, usage *OpenAIUsage) {
	if usage == nil || len(data) == 0 || bytes.Equal(bytes.TrimSpace(data), []byte("[DONE]")) {
		return
	}
	if !gjson.GetBytes(data, "usage.total_tokens").Exists() {
		return
	}
	usage.InputTokens = int(gjson.GetBytes(data, "usage.prompt_tokens").Int())
	usage.OutputTokens = int(gjson.GetBytes(data, "usage.completion_tokens").Int())
	usage.CacheReadInputTokens = int(gjson.GetBytes(data, "usage.prompt_tokens_details.cached_tokens").Int())
}

func replaceChatCompletionsModelInSSELine(line, fromModel, toModel string) string {
	data, ok := extractOpenAISSEDataLine(line)
	if !ok || data == "" || data == "[DONE]" {
		return line
	}
	if gjson.Get(data, "model").String() != fromModel {
		return line
	}
	newData, err := sjson.Set(data, "model", toModel)
	if err != nil {
		return line
	}
	return "data: " + newData
}

func replaceChatCompletionsModelInResponseBody(body []byte, fromModel, toModel string) []byte {
	if len(body) == 0 || gjson.GetBytes(body, "model").String() != fromModel {
		return body
	}
	updated, err := sjson.SetBytes(body, "model", toModel)
	if err != nil {
		return body
	}
	return updated
}
