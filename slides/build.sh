#!/bin/sh
# slides: proposal.pdf ; with speaker notes: proposal_notes.pdf
pdflatex -interaction=nonstopmode proposal.tex >/dev/null && pdflatex -interaction=nonstopmode proposal.tex >/dev/null
sed 's/^\\begin{document}/\\setbeameroption{show notes}\\begin{document}/' proposal.tex > proposal_notes.tex
pdflatex -interaction=nonstopmode proposal_notes.tex >/dev/null && pdflatex -interaction=nonstopmode proposal_notes.tex >/dev/null
rm -f *.aux *.log *.nav *.out *.snm *.toc proposal_notes.tex
