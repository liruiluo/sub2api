//go:build unit

package service

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/Wei-Shaw/sub2api/internal/pkg/tlsfingerprint"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

type queuedGatewayHTTPUpstream struct {
	responses []*http.Response
	requests  []*http.Request
	bodies    [][]byte
}

func (u *queuedGatewayHTTPUpstream) Do(req *http.Request, _ string, _ int64, _ int) (*http.Response, error) {
	if req.Body != nil {
		body, _ := io.ReadAll(req.Body)
		u.bodies = append(u.bodies, body)
		req.Body = io.NopCloser(bytes.NewReader(body))
	} else {
		u.bodies = append(u.bodies, nil)
	}
	u.requests = append(u.requests, req)
	if len(u.responses) == 0 {
		return nil, fmt.Errorf("no mocked response")
	}
	resp := u.responses[0]
	u.responses = u.responses[1:]
	return resp, nil
}

func (u *queuedGatewayHTTPUpstream) DoWithTLS(req *http.Request, proxyURL string, accountID int64, accountConcurrency int, profile *tlsfingerprint.Profile) (*http.Response, error) {
	return u.Do(req, proxyURL, accountID, accountConcurrency)
}

type compatFlagRepo struct {
	mockAccountRepoForGemini
	updatedExtra map[string]any
}

func (r *compatFlagRepo) UpdateExtra(_ context.Context, _ int64, updates map[string]any) error {
	r.updatedExtra = updates
	return nil
}

func TestOpenAIGatewayService_ForwardResponsesCompatFallback(t *testing.T) {
	gin.SetMode(gin.TestMode)
	body := []byte(`{"model":"gpt-5.4","input":"Say OK","stream":false}`)
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(body))
	c.Request.Header.Set("Content-Type", "application/json")

	upstream := &queuedGatewayHTTPUpstream{responses: []*http.Response{
		newJSONResponse(http.StatusBadRequest, `{"error":{"type":"one_hub_error","message":"field Messages is required"}}`),
		newJSONResponse(http.StatusOK, `{"id":"chatcmpl_test","object":"chat.completion","created":1,"model":"gpt-5.4","choices":[{"index":0,"message":{"role":"assistant","content":"OK"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`),
	}}
	repo := &compatFlagRepo{}
	svc := &OpenAIGatewayService{cfg: &config.Config{}, httpUpstream: upstream, accountRepo: repo}
	account := &Account{ID: 131, Platform: PlatformOpenAI, Type: AccountTypeAPIKey, Concurrency: 1, Credentials: map[string]any{"api_key": "sk-test", "base_url": "https://api.daiju.live/v1"}}

	result, err := svc.Forward(context.Background(), c, account, body)
	require.NoError(t, err)
	require.NotNil(t, result)
	require.Equal(t, 1, result.Usage.InputTokens)
	require.Equal(t, 1, result.Usage.OutputTokens)
	require.Len(t, upstream.requests, 2)
	require.Equal(t, "https://api.daiju.live/v1/responses", upstream.requests[0].URL.String())
	require.Equal(t, "https://api.daiju.live/v1/chat/completions", upstream.requests[1].URL.String())
	require.True(t, account.IsOpenAIChatCompletionsCompatEnabled())
	require.NotNil(t, repo.updatedExtra)
	require.Equal(t, true, repo.updatedExtra[openAIAPIKeyChatCompletionsCompatExtraKey])
	require.Contains(t, rec.Body.String(), `"object":"response"`)
	require.Contains(t, rec.Body.String(), `"text":"OK"`)
}

func TestOpenAIGatewayService_ForwardResponsesCompatDirectWhenFlagged(t *testing.T) {
	gin.SetMode(gin.TestMode)
	body := []byte(`{"model":"gpt-5.4","input":"Say OK","stream":false}`)
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(body))
	c.Request.Header.Set("Content-Type", "application/json")

	upstream := &queuedGatewayHTTPUpstream{responses: []*http.Response{
		newJSONResponse(http.StatusOK, `{"id":"chatcmpl_test","object":"chat.completion","created":1,"model":"gpt-5.4","choices":[{"index":0,"message":{"role":"assistant","content":"OK"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`),
	}}
	svc := &OpenAIGatewayService{cfg: &config.Config{}, httpUpstream: upstream}
	account := &Account{ID: 131, Platform: PlatformOpenAI, Type: AccountTypeAPIKey, Concurrency: 1, Credentials: map[string]any{"api_key": "sk-test", "base_url": "https://api.daiju.live/v1"}, Extra: map[string]any{openAIAPIKeyChatCompletionsCompatExtraKey: true}}

	_, err := svc.Forward(context.Background(), c, account, body)
	require.NoError(t, err)
	require.Len(t, upstream.requests, 1)
	require.Equal(t, "https://api.daiju.live/v1/chat/completions", upstream.requests[0].URL.String())
	require.False(t, strings.Contains(rec.Body.String(), "upstream_error"))
}

func TestOpenAIGatewayService_ForwardResponsesCompatDirectNormalizesDeveloperRole(t *testing.T) {
	gin.SetMode(gin.TestMode)
	body := []byte(`{"model":"gpt-5.4","input":[{"type":"message","role":"developer","content":"keep rules"},{"type":"message","role":"user","content":"Say OK"}],"stream":false}`)
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(body))
	c.Request.Header.Set("Content-Type", "application/json")

	upstream := &queuedGatewayHTTPUpstream{responses: []*http.Response{
		newJSONResponse(http.StatusOK, `{"id":"chatcmpl_test","object":"chat.completion","created":1,"model":"gpt-5.4","choices":[{"index":0,"message":{"role":"assistant","content":"OK"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`),
	}}
	svc := &OpenAIGatewayService{cfg: &config.Config{}, httpUpstream: upstream}
	account := &Account{
		ID:          131,
		Platform:    PlatformOpenAI,
		Type:        AccountTypeAPIKey,
		Concurrency: 1,
		Credentials: map[string]any{"api_key": "sk-test", "base_url": "https://api.daiju.live/v1"},
		Extra:       map[string]any{openAIAPIKeyChatCompletionsCompatExtraKey: true},
	}

	_, err := svc.Forward(context.Background(), c, account, body)
	require.NoError(t, err)
	require.Len(t, upstream.requests, 1)
	require.Len(t, upstream.bodies, 1)
	require.Equal(t, "https://api.daiju.live/v1/chat/completions", upstream.requests[0].URL.String())
	require.Equal(t, "system", gjson.GetBytes(upstream.bodies[0], "messages.0.role").String())
	require.Equal(t, "user", gjson.GetBytes(upstream.bodies[0], "messages.1.role").String())
	require.False(t, strings.Contains(rec.Body.String(), "upstream_error"))
}

func TestOpenAIGatewayService_ForwardResponsesCompatDirectPreservesAPIKeyForwardModel(t *testing.T) {
	gin.SetMode(gin.TestMode)

	tests := []struct {
		name          string
		requestModel  string
		expectedModel string
	}{
		{
			name:          "preserves bare codex alias",
			requestModel:  "codex",
			expectedModel: "codex",
		},
		{
			name:          "preserves gpt-5 alias",
			requestModel:  "gpt-5",
			expectedModel: "gpt-5",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body := []byte(fmt.Sprintf(`{"model":%q,"input":"Say OK","stream":false}`, tt.requestModel))
			rec := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(rec)
			c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(body))
			c.Request.Header.Set("Content-Type", "application/json")

			upstream := &queuedGatewayHTTPUpstream{responses: []*http.Response{
				newJSONResponse(http.StatusOK, `{"id":"chatcmpl_test","object":"chat.completion","created":1,"model":"gpt-5.4","choices":[{"index":0,"message":{"role":"assistant","content":"OK"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`),
			}}
			svc := &OpenAIGatewayService{cfg: &config.Config{}, httpUpstream: upstream}
			account := &Account{
				ID:          131,
				Platform:    PlatformOpenAI,
				Type:        AccountTypeAPIKey,
				Concurrency: 1,
				Credentials: map[string]any{"api_key": "sk-test", "base_url": "https://api.daiju.live/v1"},
				Extra:       map[string]any{openAIAPIKeyChatCompletionsCompatExtraKey: true},
			}

			_, err := svc.Forward(context.Background(), c, account, body)
			require.NoError(t, err)
			require.Len(t, upstream.requests, 1)
			require.Len(t, upstream.bodies, 1)
			require.Equal(t, "https://api.daiju.live/v1/chat/completions", upstream.requests[0].URL.String())
			require.Equal(t, tt.expectedModel, gjson.GetBytes(upstream.bodies[0], "model").String())
		})
	}
}


func TestOpenAIGatewayService_ForwardResponsesCompatFallbackOnConvertRequestFailed(t *testing.T) {
	gin.SetMode(gin.TestMode)
	body := []byte(`{"model":"gpt-5.4","input":"Say OK","stream":false}`)
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(body))
	c.Request.Header.Set("Content-Type", "application/json")

	upstream := &queuedGatewayHTTPUpstream{responses: []*http.Response{
		newJSONResponse(http.StatusInternalServerError, `{"error":{"message":"not implemented","type":"new_api_error","code":"convert_request_failed"}}`),
		newJSONResponse(http.StatusOK, `{"id":"chatcmpl_test","object":"chat.completion","created":1,"model":"gpt-5.4","choices":[{"index":0,"message":{"role":"assistant","content":"OK"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`),
	}}
	repo := &compatFlagRepo{}
	svc := &OpenAIGatewayService{cfg: &config.Config{}, httpUpstream: upstream, accountRepo: repo}
	account := &Account{ID: 171, Platform: PlatformOpenAI, Type: AccountTypeAPIKey, Concurrency: 1, Credentials: map[string]any{"api_key": "sk-test", "base_url": "https://elysiver.h-e.top/v1"}}

	result, err := svc.Forward(context.Background(), c, account, body)
	require.NoError(t, err)
	require.NotNil(t, result)
	require.Equal(t, 1, result.Usage.InputTokens)
	require.Equal(t, 1, result.Usage.OutputTokens)
	require.Len(t, upstream.requests, 2)
	require.Equal(t, "https://elysiver.h-e.top/v1/responses", upstream.requests[0].URL.String())
	require.Equal(t, "https://elysiver.h-e.top/v1/chat/completions", upstream.requests[1].URL.String())
	require.True(t, account.IsOpenAIChatCompletionsCompatEnabled())
	require.NotNil(t, repo.updatedExtra)
	require.Equal(t, true, repo.updatedExtra[openAIAPIKeyChatCompletionsCompatExtraKey])
	require.Contains(t, rec.Body.String(), `"object":"response"`)
	require.Contains(t, rec.Body.String(), `"text":"OK"`)
}
