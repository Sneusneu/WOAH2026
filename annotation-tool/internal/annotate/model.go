package annotate

import (
	"fmt"
	"math/rand"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/progress"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

var (
	headerStyle   = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("12"))
	labelStyle    = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("11"))
	questionStyle = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("10"))
	hintStyle     = lipgloss.NewStyle().Faint(true)
	emailIDStyle  = lipgloss.NewStyle().Bold(true)
	divStyle      = lipgloss.NewStyle().Foreground(lipgloss.Color("8"))
	pauseStyle    = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("14"))
	doneStyle     = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("10"))
	answerStyle   = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("13"))
)

type appScreen int

const (
	screenChunkIntro appScreen = iota
	screenAnnotating
	screenChunkPause
	screenAllDone
)

// Model is the bubbletea model for the annotation TUI.
type Model struct {
	emails    []Email
	intents   []Intent
	chunkSize int
	output    *AnnotationOutput
	outPath   string

	currentIntentIdx int
	chunk            []Email
	chunkPos         int
	screen           appScreen
	errMsg           string

	width    int
	height   int
	viewport viewport.Model
	ready    bool

	overallBar  progress.Model
	chunkBar    progress.Model
	displayTime time.Time // when the current email was first shown this session
}

// NewModel creates a new TUI model with the full scheduling logic.
func NewModel(emails []Email, intents []Intent, chunkSize int, output *AnnotationOutput, outPath string) Model {
	m := Model{
		emails:     emails,
		intents:    intents,
		chunkSize:  chunkSize,
		output:     output,
		outPath:    outPath,
		overallBar: progress.New(progress.WithDefaultGradient()),
		chunkBar:   progress.New(progress.WithDefaultGradient()),
	}
	m = m.buildNextChunk()
	return m
}

func (m Model) isAnswered(emailID, intentName string) bool {
	ann, ok := m.output.Annotations[emailID]
	if !ok {
		return false
	}
	_, answered := ann.Intents[intentName]
	return answered
}

// buildNextChunk picks the least-represented incomplete intent (fewest answered
// emails; ties broken by intents-file order) and populates m.chunk.
// Sets screenAllDone when nothing remains.
func (m Model) buildNextChunk() Model {
	bestIdx := -1
	bestCount := -1
	for i, intent := range m.intents {
		answered := 0
		hasUnanswered := false
		for _, email := range m.emails {
			if m.isAnswered(email.ID, intent.Name) {
				answered++
			} else {
				hasUnanswered = true
			}
		}
		if !hasUnanswered {
			continue
		}
		if bestIdx == -1 || answered < bestCount {
			bestIdx = i
			bestCount = answered
		}
	}

	if bestIdx == -1 {
		m.screen = screenAllDone
		return m
	}

	m.currentIntentIdx = bestIdx
	intent := m.intents[bestIdx]
	var chunk []Email
	for _, email := range m.emails {
		if !m.isAnswered(email.ID, intent.Name) {
			chunk = append(chunk, email)
			if len(chunk) >= m.chunkSize {
				break
			}
		}
	}
	rand.Shuffle(len(chunk), func(i, j int) { chunk[i], chunk[j] = chunk[j], chunk[i] })
	m.chunk = chunk
	m.chunkPos = 0
	m.screen = screenChunkIntro
	return m
}

func (m Model) currentIntent() Intent {
	return m.intents[m.currentIntentIdx]
}

func (m Model) currentEmail() Email {
	return m.chunk[m.chunkPos]
}

// IsAllDone reports whether there is no work left (useful for pre-launch check).
func (m Model) IsAllDone() bool { return m.screen == screenAllDone }

func (m Model) Init() tea.Cmd {
	return nil
}

func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		m = m.relayout()
		return m, nil

	case progress.FrameMsg:
		var cmds []tea.Cmd
		om, oCmd := m.overallBar.Update(msg)
		m.overallBar = om.(progress.Model)
		cmds = append(cmds, oCmd)
		cm, cCmd := m.chunkBar.Update(msg)
		m.chunkBar = cm.(progress.Model)
		cmds = append(cmds, cCmd)
		return m, tea.Batch(cmds...)

	case tea.KeyMsg:
		switch m.screen {
		case screenAnnotating:
			switch msg.String() {
			case "y", "Y":
				return m.recordAnswer(true)
			case "n", "N":
				return m.recordAnswer(false)
			case "s", "S":
				return m.skipEmail()
			case "left":
				if m.chunkPos > 0 {
					m.chunkPos--
					m = m.refreshViewport()
				}
				return m, nil
			case "right":
				if m.chunkPos < len(m.chunk)-1 {
					m.chunkPos++
					m = m.refreshViewport()
				}
				return m, nil
			case "q", "Q", "ctrl+c":
				_ = Save(m.outPath, m.output)
				return m, tea.Quit
			default:
				// All other keys (↑/↓/j/k/PgUp/PgDn etc.) go to the viewport.
				var cmd tea.Cmd
				m.viewport, cmd = m.viewport.Update(msg)
				return m, cmd
			}
		case screenChunkIntro:
			switch msg.String() {
			case "enter":
				m.displayTime = time.Now()
				m.screen = screenAnnotating
				m = m.relayout()
				if m.ready {
					m.viewport.GotoTop()
				}
				return m, nil
			case "q", "Q", "ctrl+c":
				_ = Save(m.outPath, m.output)
				return m, tea.Quit
			}
		case screenChunkPause:
			switch msg.String() {
			case "enter":
				m = m.buildNextChunk()
				return m, nil
			case "q", "Q", "ctrl+c":
				return m, tea.Quit
			}
		case screenAllDone:
			return m, tea.Quit
		}
	}
	return m, nil
}

// refreshViewport updates viewport content and scrolls to top after navigation.
// It also resets displayTime so that timing for this position starts fresh
// (the first-answer guard in recordAnswer prevents overwriting an existing time).
func (m Model) refreshViewport() Model {
	m.displayTime = time.Now()
	if m.ready {
		m.viewport.SetContent(m.emailContent())
		m.viewport.GotoTop()
	}
	return m
}

// relayout recalculates the viewport height to fill the space between the
// pinned top and bottom panels. Safe to call before the first WindowSizeMsg
// (width/height == 0 → no-op).
func (m Model) relayout() Model {
	if m.screen != screenAnnotating || m.width == 0 || m.height == 0 {
		return m
	}

	top := m.renderTop()
	bottom := m.renderBottom()
	topH := lipgloss.Height(top)
	bottomH := lipgloss.Height(bottom)
	vpH := m.height - topH - bottomH
	if vpH < 1 {
		vpH = 1
	}

	barW := m.width - 20
	if barW < 10 {
		barW = 10
	}
	m.overallBar.Width = barW
	m.chunkBar.Width = barW

	if !m.ready {
		m.viewport = viewport.New(m.width, vpH)
		m.ready = true
	} else {
		m.viewport.Width = m.width
		m.viewport.Height = vpH
	}
	m.viewport.SetContent(m.emailContent())
	return m
}

func (m Model) recordAnswer(answer bool) (tea.Model, tea.Cmd) {
	email := m.currentEmail()
	intent := m.currentIntent()

	ann, ok := m.output.Annotations[email.ID]
	if !ok {
		ann = EmailAnnotation{
			Intents:   make(map[string]bool),
			TimeSpent: make(map[string]float64),
		}
	} else {
		// Resumed annotations may have nil sub-maps when loaded from JSON.
		if ann.Intents == nil {
			ann.Intents = make(map[string]bool)
		}
		if ann.TimeSpent == nil {
			ann.TimeSpent = make(map[string]float64)
		}
	}
	// Capture timing only on the first answer; revisits must not overwrite it.
	if _, alreadyTimed := ann.TimeSpent[intent.Name]; !alreadyTimed {
		ann.TimeSpent[intent.Name] = time.Since(m.displayTime).Seconds()
	}
	ann.Intents[intent.Name] = answer
	m.output.Annotations[email.ID] = ann

	m.chunkPos++
	if m.chunkPos >= len(m.chunk) {
		if err := Save(m.outPath, m.output); err != nil {
			m.errMsg = fmt.Sprintf("save error: %v", err)
			return m, tea.Quit
		}
		m.screen = screenChunkPause
		return m, nil
	}

	m.displayTime = time.Now()
	if m.ready {
		m.viewport.SetContent(m.emailContent())
		m.viewport.GotoTop()
	}
	return m, nil
}

// skipEmail advances to the next email without recording any answer, so the
// pair resurfaces in a later chunk/session.
func (m Model) skipEmail() (tea.Model, tea.Cmd) {
	m.chunkPos++
	if m.chunkPos >= len(m.chunk) {
		if err := Save(m.outPath, m.output); err != nil {
			m.errMsg = fmt.Sprintf("save error: %v", err)
			return m, tea.Quit
		}
		m.screen = screenChunkPause
		return m, nil
	}

	m.displayTime = time.Now()
	if m.ready {
		m.viewport.SetContent(m.emailContent())
		m.viewport.GotoTop()
	}
	return m, nil
}

func (m Model) countAnswered() int {
	count := 0
	for _, ann := range m.output.Annotations {
		count += len(ann.Intents)
	}
	return count
}

// countChunkAnswered returns how many emails in the current chunk already have
// an answer recorded for the current intent.
func (m Model) countChunkAnswered() int {
	if m.currentIntentIdx >= len(m.intents) {
		return 0
	}
	intent := m.currentIntent()
	count := 0
	for _, email := range m.chunk {
		if m.isAnswered(email.ID, intent.Name) {
			count++
		}
	}
	return count
}

// emailContent formats the current email for display inside the viewport.
func (m Model) emailContent() string {
	email := m.currentEmail()
	w := m.width
	if w <= 0 {
		w = 80
	}
	var sb strings.Builder
	sb.WriteString(emailIDStyle.Render(fmt.Sprintf("[%s] (%s)", email.ID, email.Source)))
	sb.WriteString("\n\n")
	sb.WriteString(wordWrap(email.Text, w))
	return sb.String()
}

// wordWrap breaks plain-text lines that exceed width characters at word
// boundaries. Hard newlines in the original text are preserved.
func wordWrap(text string, width int) string {
	if width <= 0 {
		return text
	}
	var out strings.Builder
	lines := strings.Split(text, "\n")
	for i, line := range lines {
		if i > 0 {
			out.WriteByte('\n')
		}
		if len(line) <= width {
			out.WriteString(line)
			continue
		}
		words := strings.Fields(line)
		if len(words) == 0 {
			continue
		}
		col := 0
		for j, word := range words {
			wl := len(word)
			if j == 0 {
				out.WriteString(word)
				col = wl
			} else if col+1+wl > width {
				out.WriteByte('\n')
				out.WriteString(word)
				col = wl
			} else {
				out.WriteByte(' ')
				out.WriteString(word)
				col += 1 + wl
			}
		}
	}
	return out.String()
}

// divider returns a horizontal rule spanning the full terminal width.
func (m Model) divider() string {
	w := m.width
	if w == 0 {
		w = 80
	}
	return divStyle.Render(strings.Repeat("─", w))
}

// renderTop builds the pinned intent panel (name + description + examples + divider).
func (m Model) renderTop() string {
	intent := m.currentIntent()
	w := m.width
	if w <= 0 {
		w = 80
	}
	wrap := func(s string) string { return wordWrap(s, w) }
	wrapIndented := func(s string) string { return wordWrap(s, w-4) }

	var sb strings.Builder

	sb.WriteString(headerStyle.Render("INTENT: " + intent.Name))
	sb.WriteString("\n")
	sb.WriteString(wrap(intent.Description))
	sb.WriteString("\n\n")

	if len(intent.Criteria) > 0 {
		sb.WriteString(labelStyle.Render("Criteria:"))
		sb.WriteString("\n")
		for _, c := range intent.Criteria {
			sb.WriteString("  • " + wrapIndented(c) + "\n")
		}
		sb.WriteString("\n")
	}

	if len(intent.PositiveExamples) > 0 {
		sb.WriteString(labelStyle.Render("Positive examples:"))
		sb.WriteString("\n")
		for _, ex := range intent.PositiveExamples {
			sb.WriteString("  + " + wrapIndented(ex) + "\n")
		}
	}
	if len(intent.NegativeExamples) > 0 {
		sb.WriteString(labelStyle.Render("Negative examples:"))
		sb.WriteString("\n")
		for _, ex := range intent.NegativeExamples {
			sb.WriteString("  - " + wrapIndented(ex) + "\n")
		}
	}

	sb.WriteString(m.divider())
	return sb.String()
}

// renderBottom builds the pinned question + progress bars + key-hint footer.
func (m Model) renderBottom() string {
	var sb strings.Builder
	sb.WriteString(m.divider())
	sb.WriteString("\n")

	// Question line — show current answer if this email was already answered.
	question := questionStyle.Render("Does this email exhibit the intent above?")
	if m.currentIntentIdx < len(m.intents) {
		email := m.currentEmail()
		intent := m.currentIntent()
		if ann, ok := m.output.Annotations[email.ID]; ok {
			if val, ok2 := ann.Intents[intent.Name]; ok2 {
				label := "NO"
				if val {
					label = "YES"
				}
				question += "  " + answerStyle.Render("[current: "+label+"]")
			}
		}
	}
	sb.WriteString(question)
	sb.WriteString("\n")

	// Progress bars.
	totalPairs := len(m.emails) * len(m.intents)
	answeredTotal := m.countAnswered()
	answeredChunk := m.countChunkAnswered()
	chunkLen := len(m.chunk)

	overallPct := 0.0
	if totalPairs > 0 {
		overallPct = float64(answeredTotal) / float64(totalPairs)
	}
	chunkPct := 0.0
	if chunkLen > 0 {
		chunkPct = float64(answeredChunk) / float64(chunkLen)
	}

	sb.WriteString(fmt.Sprintf("  Overall: %s %d/%d\n",
		m.overallBar.ViewAs(overallPct), answeredTotal, totalPairs))
	sb.WriteString(fmt.Sprintf("  Chunk:   %s %d/%d\n",
		m.chunkBar.ViewAs(chunkPct), answeredChunk, chunkLen))

	sb.WriteString(hintStyle.Render("  y = yes   n = no   s = skip   ← / → = navigate   ↑/↓  j/k  PgUp/PgDn = scroll   q = quit"))
	return sb.String()
}

func (m Model) View() string {
	if m.errMsg != "" {
		return "Error: " + m.errMsg + "\n"
	}
	switch m.screen {
	case screenChunkIntro:
		return m.viewChunkIntro()
	case screenChunkPause:
		return m.viewChunkPause()
	case screenAllDone:
		return m.viewAllDone()
	default:
		return m.viewAnnotating()
	}
}

func (m Model) viewAnnotating() string {
	if !m.ready {
		return "Initializing…\n"
	}
	return m.renderTop() + "\n" + m.viewport.View() + "\n" + m.renderBottom()
}

func (m Model) viewChunkIntro() string {
	intent := m.intents[m.currentIntentIdx]

	w := m.width
	if w == 0 {
		w = 80
	}
	h := m.height
	if h == 0 {
		h = 24
	}

	// border (2) + padding left+right (8) + some breathing room
	contentW := w - 20
	if contentW < 20 {
		contentW = 20
	}

	wrap := func(s string) string { return wordWrap(s, contentW) }
	wrapIndented := func(s string) string { return wordWrap(s, contentW-4) }

	var contentSb strings.Builder
	contentSb.WriteString(pauseStyle.Render("Next intent to annotate") + "\n\n")
	contentSb.WriteString(labelStyle.Render("Name:") + "  " + wrap(intent.Name) + "\n\n")
	contentSb.WriteString(labelStyle.Render("Description:") + "\n" + wrap(intent.Description) + "\n\n")

	if len(intent.Criteria) > 0 {
		contentSb.WriteString(labelStyle.Render("Criteria:") + "\n")
		for _, c := range intent.Criteria {
			contentSb.WriteString("  • " + wrapIndented(c) + "\n")
		}
		contentSb.WriteString("\n")
	}

	if len(intent.PositiveExamples) > 0 {
		contentSb.WriteString(labelStyle.Render("Positive examples:") + "\n")
		for _, ex := range intent.PositiveExamples {
			contentSb.WriteString("  + " + wrapIndented(ex) + "\n")
		}
		contentSb.WriteString("\n")
	}
	if len(intent.NegativeExamples) > 0 {
		contentSb.WriteString(labelStyle.Render("Negative examples:") + "\n")
		for _, ex := range intent.NegativeExamples {
			contentSb.WriteString("  - " + wrapIndented(ex) + "\n")
		}
		contentSb.WriteString("\n")
	}

	contentSb.WriteString(fmt.Sprintf("Chunk size: %d email(s)\n\n", len(m.chunk)))
	contentSb.WriteString(hintStyle.Render("Enter = start   q = quit"))

	box := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color("12")).
		Padding(1, 4).
		Width(contentW).
		Render(contentSb.String())

	return lipgloss.Place(w, h, lipgloss.Center, lipgloss.Center, box)
}

func (m Model) viewChunkPause() string {
	intent := m.intents[m.currentIntentIdx]
	answered := m.countAnswered()
	total := len(m.emails) * len(m.intents)

	inner := pauseStyle.Render("✓  Chunk complete") +
		"\n\n" +
		fmt.Sprintf("Intent:  %s", intent.Name) +
		"\n" +
		fmt.Sprintf("Progress: %d / %d", answered, total) +
		"\n\n" +
		hintStyle.Render("Enter = continue   q = quit")

	box := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color("14")).
		Padding(1, 4).
		Render(inner)

	w, h := m.width, m.height
	if w == 0 {
		w = 80
	}
	if h == 0 {
		h = 24
	}
	return lipgloss.Place(w, h, lipgloss.Center, lipgloss.Center, box)
}

func (m Model) viewAllDone() string {
	var sb strings.Builder
	sb.WriteString(doneStyle.Render("All done! Thank you for annotating."))
	sb.WriteString("\n\n")
	sb.WriteString(hintStyle.Render("Press any key to exit."))
	sb.WriteString("\n")
	return sb.String()
}
