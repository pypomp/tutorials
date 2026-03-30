default: slides.pdf notes.pdf exercises_notes.pdf

slides.pdf : source.qmd
	quarto render "source.qmd" --to beamer

notes.pdf : source.qmd
	quarto render "source.qmd" --to pdf

exercises_notes.pdf : lesson5_exercises.qmd
	quarto render "lesson5_exercises.qmd" --to pdf
