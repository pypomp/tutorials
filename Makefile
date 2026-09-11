# Master Makefile for pypomp tutorials, courses, and manuscripts

TUTORIAL_DIRS = intro dhaka big_measles
COURSE_DIRS   = sbied
ARTICLE_DIRS  = article_jmlr article

SUBDIRS = $(TUTORIAL_DIRS) $(COURSE_DIRS) $(ARTICLE_DIRS)

.PHONY: all default html pdf tutorials sbied articles $(SUBDIRS) clean sync sync_git .venv

# Default: render all tutorials, course materials, and manuscripts
default: all

all: tutorials sbied articles

# Groupings
html: tutorials
pdf: sbied articles

# Standalone HTML tutorials
tutorials: $(TUTORIAL_DIRS)

intro:
	$(MAKE) -C intro

dhaka:
	$(MAKE) -C dhaka

big_measles:
	$(MAKE) -C big_measles

# SBIED Short Course (all chapters and polio case study)
sbied:
	$(MAKE) -C sbied

# Manuscripts
articles: $(ARTICLE_DIRS)

article_jmlr:
	$(MAKE) -C article_jmlr

article:
	$(MAKE) -C article

# Clean rendered documents and cache directories across all projects
clean:
	@for dir in $(SUBDIRS); do \
		if [ -f $$dir/Makefile ] || [ -f $$dir/makefile ]; then \
			$(MAKE) -C $$dir clean; \
		fi \
	done
	rm -rf .quarto

# Virtual environment & dependency management via uv
sync:
	uv sync

sync_git:
	uv add --git https://github.com/pypomp/pypomp.git pypomp

.venv:
	uv venv .venv --python 3.14
