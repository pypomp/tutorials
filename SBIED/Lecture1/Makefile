default: slides.pdf notes.pdf

slides.pdf : source.qmd
	quarto render "source.qmd" --to beamer

notes.pdf : source.qmd
	quarto render "source.qmd" --to pdf

