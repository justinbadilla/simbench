.PHONY: dev test

dev:
	./scripts/dev.sh

test:
	./.venv/bin/pytest
