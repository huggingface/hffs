.PHONY: quality style test

check_dirs := tests src setup.py

# Check that source code meets quality standards

quality:
	black --check $(check_dirs)
	isort --check-only $(check_dirs)
	flake8 $(check_dirs)

# Format source code automatically

style:
	black $(check_dirs)
	isort $(check_dirs)

# Run tests for the library

test:
	python -m pytest -sv ./tests/
