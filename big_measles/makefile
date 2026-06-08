# Makefile for measles jobs

# Declare phony targets (targets that don't represent files)
.PHONY: gpu clean

# Default run level if not specified
RUN_LEVEL ?= 1

# Set wall time based on run level
# Level 1: 30 seconds, Level 2: 4 minutes, Level 3: 4 hours, Level 4: 12 hours
ifeq ($(RUN_LEVEL),1)
WALL_TIME := 00:20:00
else ifeq ($(RUN_LEVEL),2)
WALL_TIME := 02:00:00
else ifeq ($(RUN_LEVEL),3)
WALL_TIME := 08:00:00
else ifeq ($(RUN_LEVEL),4)
WALL_TIME := 60:00:00
else
$(warning Unknown RUN_LEVEL '$(RUN_LEVEL)', defaulting to 12:00:00)
WALL_TIME := 60:00:00
endif

# GPU job
gpu:
	mkdir -p results/logs
	RUN_LEVEL=$(RUN_LEVEL) USE_CPU=false sbatch --time=$(WALL_TIME) big_measles.sbat

report:
	cd ../../../ && quarto render lab/model_001b/search_07/report.qmd

report_sbat:
	sbatch report.sbat
