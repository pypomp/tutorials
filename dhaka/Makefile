default: html-docs

# Section 6 shows the Dhaka global search results from the dmop paper
# repository, pinned by the dmop/ submodule. To use newer results, check out a
# new commit in dmop/, run `make figures`, and commit the updated submodule
# pointer along with the PNGs.
FIGURES = precise_raincloud precise_traces precise_raincloud_long \
          precise_traces_long precise_density_long
PNGS = $(FIGURES:%=imgs/%.png)

.PHONY: default html-docs dmop figures clean

html-docs: dhaka_tutorial.html

dmop:
	git submodule update --init --depth 1 dmop

figures: $(PNGS)

# The paper only has PDFs, which HTML can't show inline.
imgs/%.png: dmop/imgs/%.pdf pdf2png.py
	uv run --with pymupdf python pdf2png.py $< $@

dmop/imgs/%.pdf: | dmop ;

dhaka_tutorial.html: $(PNGS)

%.html: %.qmd
	quarto render "$*.qmd" --to html

clean:
	rm -f *.html
	rm -rf *_files .quarto
