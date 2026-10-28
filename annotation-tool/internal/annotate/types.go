package annotate

// Email is one record from the dataset JSON.
type Email struct {
	ID     string `json:"id"`
	Source string `json:"source"`
	Text   string `json:"text"`
}

// Intent is one record from the intents catalogue JSON.
type Intent struct {
	Name             string   `json:"name"`
	Description      string   `json:"description"`
	Criteria         []string `json:"criteria"`
	PositiveExamples []string `json:"positive_examples"`
	NegativeExamples []string `json:"negative_examples"`
}

// AnnotationOutput is the top-level output JSON (§3.3).
type AnnotationOutput struct {
	SchemaVersion int                       `json:"schema_version"`
	Annotator     string                    `json:"annotator"`
	DatasetID     string                    `json:"dataset_id"`
	IntentsID     string                    `json:"intents_id"`
	Annotations   map[string]EmailAnnotation `json:"annotations"`
}

// EmailAnnotation holds per-email annotation data.
type EmailAnnotation struct {
	Intents   map[string]bool    `json:"intents"`
	TimeSpent map[string]float64 `json:"time_spent,omitempty"`
}
