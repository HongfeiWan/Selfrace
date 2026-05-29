MAIN := main
ZH_MAIN := selfplay_selfrace_zh

.PHONY: all zh clean distclean

all:
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(MAIN).tex

zh:
	latexmk -xelatex -interaction=nonstopmode -halt-on-error $(ZH_MAIN).tex

clean:
	latexmk -c $(MAIN).tex
	latexmk -c $(ZH_MAIN).tex

distclean:
	latexmk -C $(MAIN).tex
	latexmk -C $(ZH_MAIN).tex
