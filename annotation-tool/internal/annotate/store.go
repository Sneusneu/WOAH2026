package annotate

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// LoadOutput reads and parses an existing annotation output file.
// Returns (nil, nil) if the file does not exist.
func LoadOutput(path string) (*AnnotationOutput, error) {
	data, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("reading output file: %w", err)
	}
	var out AnnotationOutput
	if err := json.Unmarshal(data, &out); err != nil {
		return nil, fmt.Errorf("parsing output JSON: %w", err)
	}
	return &out, nil
}

// Save writes out atomically via a temp file + rename so the output is never
// left half-written.
func Save(path string, output *AnnotationOutput) error {
	data, err := json.MarshalIndent(output, "", "  ")
	if err != nil {
		return fmt.Errorf("marshalling output: %w", err)
	}

	dir := filepath.Dir(path)
	tmp, err := os.CreateTemp(dir, ".annotation_tmp_*")
	if err != nil {
		return fmt.Errorf("creating temp file: %w", err)
	}
	tmpName := tmp.Name()

	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		os.Remove(tmpName)
		return fmt.Errorf("writing temp file: %w", err)
	}
	if err := tmp.Close(); err != nil {
		os.Remove(tmpName)
		return fmt.Errorf("closing temp file: %w", err)
	}
	if err := os.Rename(tmpName, path); err != nil {
		os.Remove(tmpName)
		return fmt.Errorf("renaming temp file: %w", err)
	}
	return nil
}
