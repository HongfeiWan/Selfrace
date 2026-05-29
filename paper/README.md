# Elsevier journal LaTeX template

This directory contains a minimal journal manuscript built with Elsevier's
`elsarticle` LaTeX class.

## Files

- `main.tex`: manuscript entry point
- `references.bib`: BibTeX database
- `selfplay_selfrace_zh.tex`: Chinese technical manuscript based on
  `Robust Autonomy Emerges from Self-Play` with Selfrace-specific additions
- `references_zh.bib`: BibTeX database for the Chinese manuscript
- `Makefile`: build and cleanup commands

## Build

```sh
make
```

The command generates `main.pdf`.

To build the Chinese Selfrace self-play manuscript:

```sh
make zh
```

The command generates `selfplay_selfrace_zh.pdf`.

To remove auxiliary files:

```sh
make clean
```

To remove all generated files, including the PDF:

```sh
make distclean
```

## Template source

The project uses the installed TeX Live package `elsarticle`, which is the
standard Elsevier article class. Official references:

- Elsevier author LaTeX instructions: https://www.elsevier.com/researcher/author/policies-and-guidelines/latex-instructions
- CTAN `elsarticle` package: https://ctan.org/pkg/elsarticle

For a specific target journal, update `\journal{Journal Name}` in `main.tex`
and check that journal's guide for required bibliography style, review mode,
word limits, highlights, graphical abstract, and declaration sections.
