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

# Update the dependencies
.PHONY:	update
update:
	poetry	update