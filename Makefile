all: test

format: FORCE
	pyink . --exclude=docs
	isort .

install: FORCE
	pip install -e .[dev]

lint: FORCE
	pylint coix
	pyink --check --exclude=docs
	isort --check .

test: lint FORCE
	pytest -vv -n auto

docs: FORCE
	$(MAKE) -C docs html

FORCE:
