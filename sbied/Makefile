# Top-level Makefile for Simulation-Based Inference for Epidemiological Dynamics (SBIED) tutorials
# Renders all chapters. The deprecated polio case study is not rendered by default;
# run `make polio` to render it.

CHAPTERS = chapter0 chapter1 chapter2 chapter3 chapter4 chapter5 chapter6
SUBDIRS = $(CHAPTERS) polio

.PHONY: all clean $(SUBDIRS)

all: $(CHAPTERS)

$(SUBDIRS):
	$(MAKE) -C $@

clean:
	rm -f chapter*/*.pdf
	rm -rf chapter*/*_files
	rm -rf chapter*/.quarto
