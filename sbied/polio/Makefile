default: slides.pdf notes.pdf exercises_beamer.pdf

slides.pdf : source.qmd
	quarto render "source.qmd" --to beamer

notes.pdf : source.qmd
	quarto render "source.qmd" --to pdf

exercises_beamer.pdf : exercises.qmd
	quarto render "exercises.qmd" --to beamer
