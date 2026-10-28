package cmd

import (
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/spf13/cobra"

	"github.com/Sneusneu/WOAH2026/annotation-tool/internal/annotate"
)

var (
	flagDataset   string
	flagIntents   string
	flagAnnotator string
	flagOutput    string
	flagChunkSize int
	flagShowSrc   bool
)

var annotateCmd = &cobra.Command{
	Use:   "annotate",
	Short: "Annotate emails with persuasion intents",
	RunE:  runAnnotate,
}

func init() {
	annotateCmd.Flags().StringVar(&flagDataset, "dataset", "", "Dataset JSON file (required)")
	annotateCmd.Flags().StringVar(&flagIntents, "intents", "", "Intents catalogue JSON file (required)")
	annotateCmd.Flags().StringVar(&flagAnnotator, "annotator", "", "Annotator identity (required)")
	annotateCmd.Flags().StringVar(&flagOutput, "output", "", "Output JSON file (default: annotations_<annotator>.json)")
	annotateCmd.Flags().IntVar(&flagChunkSize, "chunk-size", 20, "Emails per annotation chunk")
	annotateCmd.Flags().BoolVar(&flagShowSrc, "show-source", false, "Show email source (always shown; kept for compatibility)")

	annotateCmd.MarkFlagRequired("dataset")
	annotateCmd.MarkFlagRequired("intents")
	annotateCmd.MarkFlagRequired("annotator")

	rootCmd.AddCommand(annotateCmd)
}

func runAnnotate(cmd *cobra.Command, args []string) error {
	outPath := flagOutput
	if outPath == "" {
		if err := os.MkdirAll("annotations", 0755); err != nil {
			return fmt.Errorf("creating annotations directory: %w", err)
		}
		outPath = filepath.Join("annotations", fmt.Sprintf("annotations_%s.json", flagAnnotator))
	}

	// Load data.
	emails, err := annotate.LoadEmails(flagDataset)
	if err != nil {
		return fmt.Errorf("loading dataset: %w", err)
	}
	intents, err := annotate.LoadIntents(flagIntents)
	if err != nil {
		return fmt.Errorf("loading intents: %w", err)
	}

	if err := annotate.Validate(emails, intents); err != nil {
		return err
	}

	// Compute SHA-256 hashes of the input files (Issue 2).
	datasetID, err := annotate.HashFile(flagDataset)
	if err != nil {
		return fmt.Errorf("hashing dataset: %w", err)
	}
	intentsID, err := annotate.HashFile(flagIntents)
	if err != nil {
		return fmt.Errorf("hashing intents: %w", err)
	}

	// Load or create the annotation output; enforce resume guard (Issue 2).
	output, err := annotate.LoadOutput(outPath)
	if err != nil {
		return fmt.Errorf("loading existing output: %w", err)
	}
	if output != nil {
		// Resume guard checks.
		if output.DatasetID != datasetID {
			return fmt.Errorf("resume error: dataset file has changed (stored hash %s != current %s); aborting to prevent data corruption",
				output.DatasetID, datasetID)
		}
		if output.IntentsID != intentsID {
			return fmt.Errorf("resume error: intents file has changed (stored hash %s != current %s); aborting to prevent data corruption",
				output.IntentsID, intentsID)
		}
		if output.Annotator != flagAnnotator {
			return fmt.Errorf("resume error: annotator mismatch (file was written by %q, --annotator is %q)",
				output.Annotator, flagAnnotator)
		}
	} else {
		output = &annotate.AnnotationOutput{
			SchemaVersion: 1,
			Annotator:     flagAnnotator,
			DatasetID:     datasetID,
			IntentsID:     intentsID,
			Annotations:   make(map[string]annotate.EmailAnnotation),
		}
	}

	// Build the TUI model.
	m := annotate.NewModel(emails, intents, flagChunkSize, output, outPath)

	// If everything is already annotated, report on stdout and skip the TUI (§4.4).
	if m.IsAllDone() {
		total := len(emails) * len(intents)
		fmt.Printf("All %d annotation pairs are already complete. Nothing left to do.\n", total)
		return nil
	}

	// Best-effort save on SIGINT (Issue 3).
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT)
	go func() {
		<-sigCh
		_ = annotate.Save(outPath, output)
		os.Exit(1)
	}()

	p := tea.NewProgram(m, tea.WithAltScreen())
	if _, err := p.Run(); err != nil {
		return fmt.Errorf("TUI error: %w", err)
	}

	return nil
}
