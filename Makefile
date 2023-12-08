all: test

format: FORCE
	pyink .
	isort .

install: FORCE
	pip install -e .[dev]

lint: FORCE
	pylint coix
	pyink --check .
	isort --check .

test: lint FORCE
	pytest -vv -n auto

docs: FORCE
	$(MAKE) -C docs html

FORCE:
