default: html-docs

html-docs: dhaka_tutorial.html

%.html: %.qmd
	quarto render "$*.qmd" --to html

clean:
	rm -f *.html
	rm -rf *_files .quarto
