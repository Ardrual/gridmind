SHELL := /bin/bash

.PHONY: run-api data ingest dev web web-install web-build

run-api:
	uvicorn app.main:app --reload

# Legacy target name kept; downloads PDFs from manifest into data/raw/
data:
	python -m scripts.ingest --download --manifest data/manifest.json

# Legacy target name kept; vectorizes PDFs in data/raw/ into data/chroma/
ingest:
	python -m scripts.ingest --vectorize --manifest data/manifest.json

dev:
	@echo "Starting API and web dev servers..."
	@bash -lc 'trap "kill 0" EXIT; \
	  uvicorn app.main:app --reload & \
	  cd web && ([ -d node_modules ] || npm ci || npm install) && npm run dev'

web:
	cd web && ([ -d node_modules ] || npm ci || npm install) && npm run dev

web-install:
	cd web && ([ -d node_modules ] || npm ci || npm install)

web-build:
	cd web && ([ -d node_modules ] || npm ci || npm install) && npm run build
