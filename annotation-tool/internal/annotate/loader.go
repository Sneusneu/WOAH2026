package annotate

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
)

// LoadEmails reads and parses the dataset JSON file.
func LoadEmails(path string) ([]Email, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading dataset file: %w", err)
	}
	var emails []Email
	if err := json.Unmarshal(data, &emails); err != nil {
		return nil, fmt.Errorf("parsing dataset JSON: %w", err)
	}
	return emails, nil
}

// LoadIntents reads and parses the intents catalogue JSON file.
func LoadIntents(path string) ([]Intent, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading intents file: %w", err)
	}
	var intents []Intent
	if err := json.Unmarshal(data, &intents); err != nil {
		return nil, fmt.Errorf("parsing intents JSON: %w", err)
	}
	return intents, nil
}

// Validate checks input data for all pre-launch error conditions (§8).
// It returns a non-nil error with a clear, actionable message on any violation.
func Validate(emails []Email, intents []Intent) error {
	if len(emails) == 0 {
		return fmt.Errorf("dataset is empty: no emails to annotate")
	}
	if len(intents) == 0 {
		return fmt.Errorf("intents list is empty: no intents to annotate")
	}

	seenEmailIDs := make(map[string]bool, len(emails))
	for _, e := range emails {
		if seenEmailIDs[e.ID] {
			return fmt.Errorf("duplicate dataset id %q: each email must have a unique id", e.ID)
		}
		seenEmailIDs[e.ID] = true
	}

	seenIntentNames := make(map[string]bool, len(intents))
	for _, intent := range intents {
		if intent.Name == "" {
			return fmt.Errorf("intent with empty name found: all intents must have a non-empty name")
		}
		if seenIntentNames[intent.Name] {
			return fmt.Errorf("duplicate intent name %q: each intent must have a unique name", intent.Name)
		}
		seenIntentNames[intent.Name] = true
		if len(intent.PositiveExamples) == 0 {
			return fmt.Errorf("intent %q has no positive examples: at least one is required", intent.Name)
		}
		if len(intent.NegativeExamples) == 0 {
			return fmt.Errorf("intent %q has no negative examples: at least one is required", intent.Name)
		}
	}
	return nil
}

// HashFile returns a "sha256:<hex>" digest of the raw bytes of the given file.
func HashFile(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("reading file for hash: %w", err)
	}
	sum := sha256.Sum256(data)
	return fmt.Sprintf("sha256:%x", sum), nil
}
