default: slides.pdf notes.pdf

slides.pdf : lesson5.qmd
	quarto render "lesson5.qmd" --to beamer

notes.pdf : lesson5.qmd
	quarto render "lesson5.qmd" --to pdf
