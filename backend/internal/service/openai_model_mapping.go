package service

// resolveOpenAIForwardModel determines the upstream model for OpenAI-compatible
// forwarding. Group-level default mapping only applies when the account itself
// did not match any explicit model_mapping rule.
func resolveOpenAIForwardModel(account *Account, requestedModel, defaultMappedModel string) string {
	if account == nil {
		if defaultMappedModel != "" {
			return defaultMappedModel
		}
		return requestedModel
	}

	mappedModel := account.GetMappedModel(requestedModel)
	if !account.HasExplicitModelMapping(requestedModel) && defaultMappedModel != "" {
		return defaultMappedModel
	}
	return mappedModel
}
