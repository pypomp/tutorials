
PANDOC = pandoc
MD_FILES = $(wildcard *.md)
HTML_FILES = $(MD_FILES:.md=.html)

all: $(HTML_FILES)

%.html: %.md
	$(PANDOC) $< -o $@

clean:
	rm -f $(HTML_FILES)

