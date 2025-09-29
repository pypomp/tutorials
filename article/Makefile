default: pdf-docs

pdf-docs: ms.pdf

PDFLATEX = pdflatex
BIBTEX = bibtex
MAKEIDX = makeindex
RM = rm -f
CP = cp

REXE = R --vanilla
RCMD = $(REXE) CMD
RSCRIPT = Rscript --vanilla

VENV_DIR = .venv
PYTHON = python3.12
PIP = $(VENV_DIR)/bin/pip
VENV_PYTHON = $(VENV_DIR)/bin/python
SOURCE = source

.PHONY: venv

%.pdf: %.tex
	$(PDFLATEX) $*
	-$(BIBTEX) $*
	$(PDFLATEX) $*
	$(PDFLATEX) $*

%.pdf: %.qmd
	quarto render "$*.qmd" --to jss-pdf

%.tex: %.qmd
	quarto render "$*.qmd" --to latex

%.so: %.c
	$(RCMD) SHLIB -o $*.so $*.c
	$(RM) $*.o

%.tex: %.Rnw
	$(RSCRIPT) -e "library(knitr); knit(\"$*.Rnw\")"

%.R: %.Rnw
	$(RSCRIPT) -e "library(knitr); purl(\"$*.Rnw\")"

%.idx: %.tex
	-$(LATEX) $*

%.ind: %.idx
	$(MAKEIDX) $*

%.pdf: %.eps
	$(EPSTOPDF) $*.eps --outfile=$*.pdf

clean:
	$(RM) *.log *.blg *.ilg *.aux *.lof *.lot *.toc *.idx
	$(RM) *.ttt *.fff *.out *.nav *.snm 
	$(RM) *.o *.so
	$(RM) *.brf
	$(RM) Rplots.ps Rplots.pdf

fresh: clean
	$(RM) *.ps *.bbl *.ind *.dvi 
	$(RM) -r cache/* figure/*


# Create a virtual Python environment
venv:
	@echo "Creating virtual environment in $(VENV_DIR)..."
	$(PYTHON) -m venv $(VENV_DIR)
	@echo "Virtual environment created."
	@echo "Installing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "Dependencies installed."



