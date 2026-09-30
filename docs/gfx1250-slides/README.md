# CDNA5 / gfx1250 programming slides

Marp draft of the [reviewed plan](../Gfx1250ProgrammingSlidesPlan.md):
12 main content slides for 30 minutes, two section dividers, and nine backup
slides (23 slides total). The dividers introduce the grouped GEMM case study
(slide 6) and the backup material (slide 14).

[Presenter transcript](transcript.md): spoken script for every slide, with timing
windows, delivery cues, and optional backup explanations.

## Build locally

Requires Node.js 20+ and npm. No global npm packages are required.

```bash
cd docs/gfx1250-slides
npm ci
npm run browser:install
npm run build
```

The browser command downloads a pinned Chrome Headless Shell into this directory's
`.cache/browser/`. Skip it if a supported browser is already installed; set
`CHROME_PATH` to select an executable explicitly. Browser system libraries must
be available on the host.

On hosts that disable unprivileged browser sandboxes (including the draft build
host), export this trusted local deck with:

```bash
CHROME_NO_SANDBOX=1 npm run build
```

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
- `package.json` and `package-lock.json`: pinned local build dependencies.

In the Marp VS Code extension, enable HTML and select `theme.css` as a custom
theme. Alternatively, `npm run preview` opens the CLI preview when a full browser
is available.

All assembly panels are filled and assembler-checked. Slides 2, 3, 5, 8, and 10 use
generated grouped-GEMM excerpts; backups B/D/I use handwritten ISA
examples. [Assembly preparation records](assembly.md) describe their provenance,
assumptions, and reproduction using `scripts/compile-assembly.py`.

XDL efficiency cells and the resource-comparison panel remain draft placeholders.
Search for `TODO` in `slides.md` to find their preparation notes. XDL results use percentages and
percentage-point changes, with full-kernel and steady-loop metrics kept separate.
The draft contains no absolute performance results or invented chart values.

The code reference is `b266fe4c1d`; ISA sections refer to the AMD CDNA5 Reference
Guide dated 27 July 2026. Technical citations are on slides, with further sources
and preparation details in the plan. Collection details are outside the deck.
