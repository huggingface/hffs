.PHONY: quality style test

# Check that source code meets quality standards

quality:
	black --check --line-length 119 --target-version py37 src tests
	isort --check-only src tests
	flake8 src tests

# Format source code automatically

style:
	black --line-length 119 --target-version py37 src tests
	isort src tests

# Run tests for the library

test:
	python -m pytest -sv ./tests/
