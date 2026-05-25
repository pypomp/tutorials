# Top-level Makefile for Simulation-Based Inference for Epidemiological Dynamics (SBIED) tutorials
# Renders all chapters and the polio case study.

SUBDIRS = chapter0 chapter1 chapter2 chapter3 chapter4 chapter5 chapter6 polio

.PHONY: all clean $(SUBDIRS)

all: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@

clean:
	rm -f chapter*/*.pdf polio/*.pdf
	rm -rf chapter*/*_files polio/*_files
	rm -rf chapter*/.quarto polio/.quarto
