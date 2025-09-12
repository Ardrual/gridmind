run-api:
	uvicorn app.main:app --reload

data:
	python -m scripts.ingest --download --verify --manifest data/manifest.json

ingest:
	python -m scripts.ingest --chunk --embed --manifest data/manifest.json

dev:
	make -j2 run-api web

web:
	cd web && npm run dev
