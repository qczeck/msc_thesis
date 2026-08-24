JOB      = final_report
CHAPTERS = Titlepage Introduction Background RoPART Learning MatchedBudget Downstream Conclusion Declarations Appendix

# latexmk reruns LaTeX until the references settle and calls biber itself.
all:
	latexmk -synctex=1 $(JOB).tex

# build a single chapter, e.g. `make Introduction`
% : %.tex
	latexmk -synctex=1 $<

# `latexmk -C` only knows about the job you name, so sweep the per-chapter
# artefacts too: a standalone chapter build leaves its own .bcf/.bbl behind,
# and latexmk will try to run biber on any stray .bcf it finds next time.
clean:
	latexmk -C $(JOB).tex
	for f in $(CHAPTERS); do latexmk -C $$f.tex; done
	rm -f $(JOB).bbl $(JOB).run.xml
	rm -f $(addsuffix .bbl,$(CHAPTERS)) $(addsuffix .run.xml,$(CHAPTERS))

.PHONY: all clean
