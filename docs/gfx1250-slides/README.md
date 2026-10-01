# CDNA5 / gfx1250 programming slides

Marp draft of the [reviewed plan](../Gfx1250ProgrammingSlidesPlan.md):
one title slide, 12 main content slides, two section dividers, and ten backup
slides (25 slides total). Slides 2–6 cover hardware. The dividers introduce the
grouped GEMM case study (slide 7) and the backup material (slide 15).
The main talk runs for 30 minutes, including five minutes for discussion.

[Presenter transcript](transcript.md): spoken script for every slide, with timing
windows, delivery cues, and optional backup explanations.

[Bullet-only presenter outline](transcript-bullets.md): short cues for all 25 slides,
with main-talk timing and optional backup reminders.

## Build locally

Requires Bun 1.3+ and Firefox. Use Bun for dependencies and export.

```bash
cd docs/gfx1250-slides
bun install --frozen-lockfile
bun run build
```

The build uses the installed Firefox. It creates a temporary browser profile
under the home directory and removes it when the build exits.

Outputs:

- `dist/gfx1250-draft.pdf`: complete deck, including backup slides.
- `dist/gfx1250-draft.html`: browser presentation with speaker notes.

Both exports embed the SVG diagrams and can be moved as standalone files.
`node_modules/`, `.cache/`, and generated `dist/` files are ignored by Git.

## Edit

- `slides.md`: slide text and speaker notes in HTML comments; `---` separates slides.
- `transcript.md`: full spoken script for the main talk and all backup slides.
- `theme.css`: typography, colors, layout, and placeholder styling.
- `assets/*.svg`: editable diagrams; pipeline diagrams are schematic.
- `package.json` and `bun.lock`: pinned local build dependencies.

In the Marp VS Code extension, enable HTML and select `theme.css` as a custom
theme. Alternatively, `bun run preview` opens the CLI preview when a full browser
is available.

All assembly panels are filled and assembler-checked. Slides 3, 4, 6, 10, and 12 use
generated grouped-GEMM excerpts; backups B/D/I use handwritten ISA
examples. [Assembly preparation records](assembly.md) describe their provenance,
assumptions, and reproduction using `scripts/compile-assembly.py`.

[XDL efficiency preparation records](efficiency.md) define the controlled
five-variant comparison, steady-state metric, and collection procedure.
`scripts/collect-efficiency.py` launches one variant and checks every output
against a CPU reference in a dedicated scratch directory.

The XDL efficiency comparison slide is deferred. The [preparation record](efficiency.md)
retains all five validated results at G=2, M=4096, N=1024, K=2048, P=32,
including precise values, provenance, and links to the raw artifacts.

The resource-comparison panel in backup H remains a draft placeholder.
Search for `TODO` in `slides.md` to find its preparation notes. Retained XDL results use
percentages and percentage-point changes; steady-state and whole-kernel scopes
remain separate. The draft contains no absolute performance results.

The code reference is `b266fe4c1d`; ISA sections refer to the AMD CDNA5 Reference
Guide dated 27 July 2026. Technical citations are on slides, with further sources
and preparation details in the plan. Collection details are outside the deck.
