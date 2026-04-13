package service

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/pkg/openai"
	"github.com/stretchr/testify/require"
)

type openaiOAuthClientRefreshStub struct {
	refreshCalls int32
}

func (s *openaiOAuthClientRefreshStub) ExchangeCode(ctx context.Context, code, codeVerifier, redirectURI, proxyURL, clientID string) (*openai.TokenResponse, error) {
	return nil, errors.New("not implemented")
}

func (s *openaiOAuthClientRefreshStub) RefreshToken(ctx context.Context, refreshToken, proxyURL string) (*openai.TokenResponse, error) {
	atomic.AddInt32(&s.refreshCalls, 1)
	return nil, errors.New("not implemented")
}

func (s *openaiOAuthClientRefreshStub) RefreshTokenWithClientID(ctx context.Context, refreshToken, proxyURL string, clientID string) (*openai.TokenResponse, error) {
	atomic.AddInt32(&s.refreshCalls, 1)
	return nil, errors.New("not implemented")
}

func TestOpenAIOAuthService_RefreshAccountToken_NoRefreshTokenUsesExistingAccessToken(t *testing.T) {
	client := &openaiOAuthClientRefreshStub{}
	svc := NewOpenAIOAuthService(nil, client)

	expiresAt := time.Now().Add(30 * time.Minute).UTC().Format(time.RFC3339)
	account := &Account{
		ID:       77,
		Platform: PlatformOpenAI,
		Type:     AccountTypeOAuth,
		Credentials: map[string]any{
			"access_token": "existing-access-token",
			"expires_at":   expiresAt,
			"client_id":    "client-id-1",
		},
	}

	info, err := svc.RefreshAccountToken(context.Background(), account)
	require.NoError(t, err)
	require.NotNil(t, info)
	require.Equal(t, "existing-access-token", info.AccessToken)
	require.Equal(t, "client-id-1", info.ClientID)
	require.Zero(t, atomic.LoadInt32(&client.refreshCalls), "existing access token should be reused without calling refresh")
}

func TestOpenAIOAuthService_RefreshAccountToken_NoRefreshTokenFallsBackToSessionToken(t *testing.T) {
	client := &openaiOAuthClientRefreshStub{}
	svc := NewOpenAIOAuthService(nil, client)

	var seenCookie atomic.Value
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seenCookie.Store(r.Header.Get("Cookie"))
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"accessToken":"session-access","expires":"2030-01-02T03:04:05Z","user":{"id":"user-1","email":"test@example.com"},"account":{"id":"acct-1","planType":"pro"}}`))
	}))
	defer server.Close()

	oldURL := openAIChatGPTSessionAuthURL
	openAIChatGPTSessionAuthURL = server.URL
	defer func() { openAIChatGPTSessionAuthURL = oldURL }()

	account := &Account{
		ID:       88,
		Platform: PlatformOpenAI,
		Type:     AccountTypeOAuth,
		Credentials: map[string]any{
			"session_token": "session-token-abc",
			"cookies":       "foo=bar",
		},
	}

	info, err := svc.RefreshAccountToken(context.Background(), account)
	require.NoError(t, err)
	require.NotNil(t, info)
	require.Equal(t, "session-access", info.AccessToken)
	require.Equal(t, "acct-1", info.ChatGPTAccountID)
	require.Equal(t, "user-1", info.ChatGPTUserID)
	require.Equal(t, "pro", info.PlanType)
	require.Equal(t, "test@example.com", info.Email)
	require.Zero(t, atomic.LoadInt32(&client.refreshCalls))
	cookie, ok := seenCookie.Load().(string)
	require.True(t, ok)
	require.Contains(t, cookie, "__Secure-next-auth.session-token=session-token-abc")
	require.Contains(t, cookie, "foo=bar")
}

func TestOpenAIOAuthService_RefreshAccountToken_RefreshFailureFallsBackToSessionToken(t *testing.T) {
	client := &openaiOAuthClientRefreshStub{}
	svc := NewOpenAIOAuthService(nil, client)

	var seenCookie atomic.Value
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seenCookie.Store(r.Header.Get("Cookie"))
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"accessToken":"fallback-access","expires":"2031-02-03T04:05:06Z","user":{"id":"user-2","email":"fallback@example.com"},"account":{"id":"acct-2","planType":"plus"}}`))
	}))
	defer server.Close()

	oldURL := openAIChatGPTSessionAuthURL
	openAIChatGPTSessionAuthURL = server.URL
	defer func() { openAIChatGPTSessionAuthURL = oldURL }()

	account := &Account{
		ID:       89,
		Platform: PlatformOpenAI,
		Type:     AccountTypeOAuth,
		Credentials: map[string]any{
			"refresh_token": "refresh-token-1",
			"session_token": "__Secure-next-auth.session-token=session-token-fallback;",
		},
	}

	info, err := svc.RefreshAccountToken(context.Background(), account)
	require.NoError(t, err)
	require.NotNil(t, info)
	require.Equal(t, "fallback-access", info.AccessToken)
	require.Equal(t, "acct-2", info.ChatGPTAccountID)
	require.Equal(t, "user-2", info.ChatGPTUserID)
	require.Equal(t, "plus", info.PlanType)
	require.Equal(t, "fallback@example.com", info.Email)
	require.Equal(t, int32(1), atomic.LoadInt32(&client.refreshCalls))
	require.Contains(t, seenCookie.Load().(string), "__Secure-next-auth.session-token=session-token-fallback")
}
