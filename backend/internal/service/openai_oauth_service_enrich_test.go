package service

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"

	"github.com/imroc/req/v3"
	"github.com/stretchr/testify/require"
)

func testPrivacyClientFactory(t *testing.T, handler http.HandlerFunc) PrivacyClientFactory {
	t.Helper()

	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)

	target, err := url.Parse(server.URL)
	require.NoError(t, err)

	return func(proxyURL string) (*req.Client, error) {
		client := req.C()
		client.GetTransport().WrapRoundTripFunc(func(rt http.RoundTripper) req.HttpRoundTripFunc {
			return func(r *http.Request) (*http.Response, error) {
				clone := r.Clone(r.Context())
				clone.URL.Scheme = target.Scheme
				clone.URL.Host = target.Host
				clone.Host = target.Host
				return rt.RoundTrip(clone)
			}
		})
		return client, nil
	}
}

func TestOpenAIOAuthService_EnrichTokenInfo_UsesOrganizationMatchAndSetsPrivacy(t *testing.T) {
	t.Parallel()

	accountChecks := 0
	privacyWrites := 0
	factory := testPrivacyClientFactory(t, func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "Bearer access-token-1", r.Header.Get("Authorization"))

		switch r.URL.Path {
		case "/backend-api/accounts/check/v4-2023-04-27":
			accountChecks++
			w.Header().Set("Content-Type", "application/json")
			require.NoError(t, json.NewEncoder(w).Encode(map[string]any{
				"accounts": map[string]any{
					"org_free": map[string]any{
						"account": map[string]any{
							"plan_type": "Free",
						},
						"entitlement": map[string]any{
							"expires_at": "2026-04-01T00:00:00Z",
						},
					},
					"org_paid": map[string]any{
						"account": map[string]any{
							"plan_type": "Pro",
						},
						"entitlement": map[string]any{
							"expires_at": "2026-05-02T20:32:12+00:00",
						},
					},
				},
			}))
		case "/backend-api/settings/account_user_setting":
			privacyWrites++
			require.Equal(t, "training_allowed", r.URL.Query().Get("feature"))
			require.Equal(t, "false", r.URL.Query().Get("value"))
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"success":true}`))
		default:
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
	})

	svc := NewOpenAIOAuthService(nil, nil)
	svc.SetPrivacyClientFactory(factory)

	tokenInfo := &OpenAITokenInfo{
		AccessToken:    "access-token-1",
		OrganizationID: "org_paid",
	}

	svc.enrichTokenInfo(context.Background(), tokenInfo, "")

	require.Equal(t, "Pro", tokenInfo.PlanType)
	require.Equal(t, "2026-05-02T20:32:12+00:00", tokenInfo.SubscriptionExpiresAt)
	require.Equal(t, PrivacyModeTrainingOff, tokenInfo.PrivacyMode)
	require.Equal(t, 1, accountChecks)
	require.Equal(t, 1, privacyWrites)
}

func TestOpenAIOAuthService_EnrichTokenInfo_FallsBackToPaidAccountAndPreservesEmail(t *testing.T) {
	t.Parallel()

	accountChecks := 0
	privacyWrites := 0
	factory := testPrivacyClientFactory(t, func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "Bearer access-token-2", r.Header.Get("Authorization"))

		switch r.URL.Path {
		case "/backend-api/accounts/check/v4-2023-04-27":
			accountChecks++
			w.Header().Set("Content-Type", "application/json")
			require.NoError(t, json.NewEncoder(w).Encode(map[string]any{
				"accounts": map[string]any{
					"org_free": map[string]any{
						"account": map[string]any{
							"plan_type":  "Free",
							"is_default": false,
						},
						"entitlement": map[string]any{
							"expires_at": "2026-04-01T00:00:00Z",
						},
					},
					"org_team": map[string]any{
						"account": map[string]any{
							"plan_type":  "Team",
							"is_default": false,
						},
						"entitlement": map[string]any{
							"expires_at": "2026-08-08T08:08:08Z",
						},
					},
				},
			}))
		case "/backend-api/settings/account_user_setting":
			privacyWrites++
			w.WriteHeader(http.StatusForbidden)
			_, _ = w.Write([]byte(`cloudflare Just a moment`))
		default:
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
	})

	svc := NewOpenAIOAuthService(nil, nil)
	svc.SetPrivacyClientFactory(factory)

	tokenInfo := &OpenAITokenInfo{
		AccessToken: "access-token-2",
		Email:       "keep@example.com",
	}

	svc.enrichTokenInfo(context.Background(), tokenInfo, "")

	require.Equal(t, "Team", tokenInfo.PlanType)
	require.Equal(t, "2026-08-08T08:08:08Z", tokenInfo.SubscriptionExpiresAt)
	require.Equal(t, "keep@example.com", tokenInfo.Email)
	require.Equal(t, PrivacyModeCFBlocked, tokenInfo.PrivacyMode)
	require.Equal(t, 1, accountChecks)
	require.Equal(t, 1, privacyWrites)
}

func TestOpenAIOAuthService_EnrichTokenInfo_SkipsWhenFactoryUnavailable(t *testing.T) {
	t.Parallel()

	svc := NewOpenAIOAuthService(nil, nil)
	svc.SetPrivacyClientFactory(func(proxyURL string) (*req.Client, error) {
		return nil, errors.New("factory failed")
	})

	tokenInfo := &OpenAITokenInfo{
		AccessToken: "access-token-3",
		Email:       "keep@example.com",
	}

	svc.enrichTokenInfo(context.Background(), tokenInfo, "")

	require.Equal(t, "keep@example.com", tokenInfo.Email)
	require.Equal(t, "", tokenInfo.PlanType)
	require.Equal(t, "", tokenInfo.SubscriptionExpiresAt)
	require.Equal(t, PrivacyModeFailed, tokenInfo.PrivacyMode)
}
