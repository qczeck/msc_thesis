# pdf output by default, so a bare `latexmk` (or `latexmk -c`) does the right
# thing. latexmk detects biblatex's .bcf and runs biber, not bibtex.
$pdf_mode = 1;
$bibtex_use = 2;   # run biber/bibtex, and clean the .bbl on `latexmk -C`
