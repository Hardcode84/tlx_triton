# Presentation assets

The deck follows the corporate layouts in the user-supplied
`/home/ibutygin/amd_template.pptx`, inspected on 1 October 2026. The PowerPoint
file is a design reference; rebuilding the deck does not require it.

- `amd-title.jpg`: unmodified `ppt/media/image1.jpg`, used by layout 1,
  “Title Slide - No Image.”
- `amd-lockup.png`: unmodified `ppt/media/image2.png`, the white AMD logo with
  “together we advance_,” used on title and content slides.
- `amd-logo.png`: unmodified `ppt/media/image3.png`, the white AMD wordmark,
  used by layout 27, “Divider slide.”

These three images retain AMD's supplied artwork. The theme adapts layout 1
for the title, layout 4 for content, and layout 27 for section dividers.
Slides 31–32 and 38 provide the palette and panel references: black/white,
gray surfaces, corporate gold `#c1a968`, and product teal `#00c2de`.
The six editable SVG diagrams use those colors while retaining their labels,
dimensions, connections, and timing relationships.

The corporate theme specifies Arial. `fonts/liberation-sans-regular.woff` and
`fonts/liberation-sans-bold.woff` provide compatible metrics without a global
font installation. They are WOFF conversions of the unmodified Liberation Sans
2.1.5 Regular/Bold TrueType fonts from Ubuntu's `fonts-liberation` package
`1:2.1.5-3`; glyphs and font names are preserved. Their SIL Open Font License
and attribution are included in [fonts/LICENSE.txt](fonts/LICENSE.txt).

`scripts/build.mjs` embeds artwork and fonts into the exported theme. It also
embeds the font faces into each SVG because image documents do not inherit
the surrounding page's fonts. Both PDF and HTML therefore keep the same
typography when moved away from the source checkout.
