#################################################################################
# GLOBALS                                                                       #
#################################################################################
PYTHON	=	python
PROJECT_DIR	=	$(CURDIR)
#################################################################################
# COMMANDS                                                                      #
#################################################################################
# Makefile
POETRY_INSTALL_URL=https://install.python-poetry.org

# Define the virtual environment directory
VENV_DIR=.venv

# Install Poetry
.PHONY:	install-poetry
install-poetry:
	curl	-sSL	$(POETRY_INSTALL_URL)	|	python3	-

# Install production dependencies
.PHONY:	install
install:	install-poetry
	poetry	install

# Install development dependencies
.PHONY:	install-dev
install-dev:	install
	poetry	install	--dev

# Install debugpy for debugging
.PHONY:	install-debugpy
install-debugpy:
	poetry	add	debugpy	--dev

# Update the dependencies
.PHONY:	update
update:
	poetry	update

# Download the data to raw directory
.PHONY:	download-data
download-data:
	poetry	run	python	denoising_diffusion/dataset.py

# Activate poetry environment and run debugpy
.PHONY:	debug-train
debug-train:
	poetry	run	python	-m	debugpy	--listen	0.0.0.0:5678	--wait-for-client	denoising_diffusion/modeling/train.py