.PHONY: install run lint type test check sandbox-image

install:
	pip install -e ".[dev]"

run:
	python -m orchestration.main

lint:
	ruff check .

type:
	mypy src/orchestration/contracts.py src/orchestration/sandbox.py src/orchestration/routing.py src/orchestration/config.py

test:
	pytest -q

check: lint type test

sandbox-image:
	docker build -f Dockerfile.sandbox -t agent-sandbox:py312 .
