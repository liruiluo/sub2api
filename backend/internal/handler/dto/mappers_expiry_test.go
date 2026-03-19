package dto

import (
	"testing"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/service"
)

func TestAccountFromServiceShallow_UsesTopLevelExpiresAtFirst(t *testing.T) {
	top := time.Date(2026, 3, 21, 8, 0, 0, 0, time.UTC)
	acc := &service.Account{
		ExpiresAt: &top,
		Credentials: map[string]any{
			"expires_at": "1774257557",
		},
	}

	got := AccountFromServiceShallow(acc)
	if got == nil || got.ExpiresAt == nil {
		t.Fatal("expected expires_at to be set")
	}
	if *got.ExpiresAt != top.Unix() {
		t.Fatalf("expires_at = %d, want %d", *got.ExpiresAt, top.Unix())
	}
}

func TestAccountFromServiceShallow_FallsBackToCredentialsExpiresAt(t *testing.T) {
	acc := &service.Account{
		Credentials: map[string]any{
			"expires_at": "1774257557",
		},
	}

	got := AccountFromServiceShallow(acc)
	if got == nil || got.ExpiresAt == nil {
		t.Fatal("expected expires_at to be inferred from credentials")
	}
	if *got.ExpiresAt != 1774257557 {
		t.Fatalf("expires_at = %d, want %d", *got.ExpiresAt, int64(1774257557))
	}
}

func TestAccountFromServiceShallow_FallsBackToExtraExpiresAtRFC3339(t *testing.T) {
	ts := "2026-03-29T15:26:00+08:00"
	parsed, err := time.Parse(time.RFC3339, ts)
	if err != nil {
		t.Fatalf("parse test time: %v", err)
	}
	acc := &service.Account{
		Extra: map[string]any{
			"expires_at": ts,
		},
	}

	got := AccountFromServiceShallow(acc)
	if got == nil || got.ExpiresAt == nil {
		t.Fatal("expected expires_at to be inferred from extra")
	}
	if *got.ExpiresAt != parsed.Unix() {
		t.Fatalf("expires_at = %d, want %d", *got.ExpiresAt, parsed.Unix())
	}
}

func TestAccountFromServiceShallow_NormalizesMillisecondExpiry(t *testing.T) {
	acc := &service.Account{
		Credentials: map[string]any{
			"expires_at": float64(1774257557000),
		},
	}

	got := AccountFromServiceShallow(acc)
	if got == nil || got.ExpiresAt == nil {
		t.Fatal("expected expires_at to be inferred from millisecond timestamp")
	}
	if *got.ExpiresAt != 1774257557 {
		t.Fatalf("expires_at = %d, want %d", *got.ExpiresAt, int64(1774257557))
	}
}

func TestAccountFromServiceShallow_FallsBackToAccessTokenExp(t *testing.T) {
	acc := &service.Account{
		Credentials: map[string]any{
			"access_token": "eyJhbGciOiJub25lIn0.eyJleHAiOjE3NzQ2OTQ1MTJ9.",
		},
	}

	got := AccountFromServiceShallow(acc)
	if got == nil || got.ExpiresAt == nil {
		t.Fatal("expected expires_at to be inferred from access token exp")
	}
	if *got.ExpiresAt != 1774694512 {
		t.Fatalf("expires_at = %d, want %d", *got.ExpiresAt, int64(1774694512))
	}
}
