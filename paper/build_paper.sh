#!/usr/bin/env bash
# Rebuild the report: figures and numbers from data/, then LaTeX + BibTeX.
# Usage (from the repository root or from paper/):  bash paper/build_paper.sh
# Needs Python with numpy/matplotlib and a TeX distribution (pdflatex, bibtex).
set -euo pipefail
cd "$(dirname "$0")"

python make_figures.py

pdflatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null
bibtex main > /dev/null
pdflatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null
pdflatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null

grep -E "Warning.*(undefined|multiply)" main.log || true
rm -f main.aux main.bbl main.blg main.log main.out
echo "paper/main.pdf rebuilt"
