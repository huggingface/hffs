.PHONY: quality style test

check_dirs := tests src setup.py

# Check that source code meets quality standards

quality:
	black --check $(check_dirs)
	ruff $(check_dirs)

# Format source code automatically

style:
	black $(check_dirs)
	ruff $(check_dirs) --fix

# Run tests for the library

test:
	python -m pytest -sv ./tests/
