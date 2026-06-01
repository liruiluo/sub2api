package service

import (
	"strings"
	"testing"
)

func mustParseClaudeMaxSimulationRequest(t *testing.T, body string) *ParsedRequest {
	t.Helper()
	parsed, err := ParseGatewayRequest(NewRequestBodyRef([]byte(body)), PlatformAnthropic)
	if err != nil {
		t.Fatalf("parse request: %v", err)
	}
	return parsed
}

func TestProjectUsageToClaudeMax1H_Conservation(t *testing.T) {
	usage := &ClaudeUsage{
		InputTokens:              1200,
		CacheCreationInputTokens: 0,
		CacheCreation5mTokens:    0,
		CacheCreation1hTokens:    0,
	}
	parsed := mustParseClaudeMaxSimulationRequest(t, `{
		"model":"claude-sonnet-4-5",
		"messages":[{
			"role":"user",
			"content":[
				{"type":"text","text":"`+strings.Repeat("cached context ", 200)+`","cache_control":{"type":"ephemeral"}},
				{"type":"text","text":"summarize quickly"}
			]
		}]
	}`)

	changed := projectUsageToClaudeMax1H(usage, parsed)
	if !changed {
		t.Fatalf("expected usage to be projected")
	}

	total := usage.InputTokens + usage.CacheCreation5mTokens + usage.CacheCreation1hTokens
	if total != 1200 {
		t.Fatalf("total tokens changed: got=%d want=%d", total, 1200)
	}
	if usage.CacheCreation5mTokens != 0 {
		t.Fatalf("cache_creation_5m should be 0, got=%d", usage.CacheCreation5mTokens)
	}
	if usage.InputTokens <= 0 || usage.InputTokens >= 1200 {
		t.Fatalf("simulated input out of range, got=%d", usage.InputTokens)
	}
	if usage.InputTokens > 100 {
		t.Fatalf("simulated input should stay near cache breakpoint tail, got=%d", usage.InputTokens)
	}
	if usage.CacheCreation1hTokens <= 0 {
		t.Fatalf("cache_creation_1h should be > 0, got=%d", usage.CacheCreation1hTokens)
	}
	if usage.CacheCreationInputTokens != usage.CacheCreation1hTokens {
		t.Fatalf("cache_creation_input_tokens mismatch: got=%d want=%d", usage.CacheCreationInputTokens, usage.CacheCreation1hTokens)
	}
}

func TestComputeClaudeMaxProjectedInputTokens_Deterministic(t *testing.T) {
	parsed := mustParseClaudeMaxSimulationRequest(t, `{
		"model":"claude-opus-4-5",
		"messages":[{
			"role":"user",
			"content":[
				{"type":"text","text":"build context","cache_control":{"type":"ephemeral"}},
				{"type":"text","text":"what is failing now"}
			]
		}]
	}`)

	got1 := computeClaudeMaxProjectedInputTokens(4096, parsed)
	got2 := computeClaudeMaxProjectedInputTokens(4096, parsed)
	if got1 != got2 {
		t.Fatalf("non-deterministic input tokens: %d != %d", got1, got2)
	}
}

func TestShouldSimulateClaudeMaxUsage(t *testing.T) {
	group := &Group{
		Platform:                 PlatformAnthropic,
		SimulateClaudeMaxEnabled: true,
	}
	input := &RecordUsageInput{
		Result: &ForwardResult{
			Model: "claude-sonnet-4-5",
			Usage: ClaudeUsage{
				InputTokens:              3000,
				CacheCreationInputTokens: 0,
				CacheCreation5mTokens:    0,
				CacheCreation1hTokens:    0,
			},
		},
		ParsedRequest: mustParseClaudeMaxSimulationRequest(t, `{"messages":[{"role":"user","content":[{"type":"text","text":"cached","cache_control":{"type":"ephemeral"}},{"type":"text","text":"tail"}]}]}`),
		APIKey:        &APIKey{Group: group},
	}

	if !shouldSimulateClaudeMaxUsage(input) {
		t.Fatalf("expected simulate=true for claude group with cache signal")
	}

	input.ParsedRequest = mustParseClaudeMaxSimulationRequest(t, `{"messages":[{"role":"user","content":"no cache signal"}]}`)
	if shouldSimulateClaudeMaxUsage(input) {
		t.Fatalf("expected simulate=false when request has no cache signal")
	}

	input.ParsedRequest = mustParseClaudeMaxSimulationRequest(t, `{"messages":[{"role":"user","content":[{"type":"text","text":"cached","cache_control":{"type":"ephemeral"}}]}]}`)
	input.Result.Usage.CacheCreationInputTokens = 100
	if shouldSimulateClaudeMaxUsage(input) {
		t.Fatalf("expected simulate=false when cache creation already exists")
	}
}
