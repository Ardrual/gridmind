run-api:
	# Requires a FastAPI app in app/main.py (placeholder today)
	uvicorn app.main:app --reload

# Legacy target name kept; downloads PDFs from manifest into data/raw/
data:
	python -m scripts.ingest --download --manifest data/manifest.json

# Legacy target name kept; vectorizes PDFs in data/raw/ into data/chroma/
ingest:
	python -m scripts.ingest --vectorize --manifest data/manifest.json

dev:
	make -j2 run-api web

web:
	cd web && npm run dev
