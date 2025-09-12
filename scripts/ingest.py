
import os
import json
import requests
import argparse

def download_manifest_pdfs():
	"""Download all PDFs listed in data/manifest.json into data/raw."""
	MANIFEST_PATH = os.path.join('data', 'manifest.json')
	RAW_DIR = os.path.join('data', 'raw')
	os.makedirs(RAW_DIR, exist_ok=True)
	with open(MANIFEST_PATH, 'r') as f:
		manifest = json.load(f)
	for entry in manifest:
		url = entry['url']
		file_id = entry['id']
		filename = f"{file_id}.pdf"
		out_path = os.path.join(RAW_DIR, filename)
		print(f"Downloading {url} -> {out_path}")
		try:
			response = requests.get(url, stream=True)
			response.raise_for_status()
			with open(out_path, 'wb') as f_out:
				for chunk in response.iter_content(chunk_size=8192):
					if chunk:
						f_out.write(chunk)
			print(f"Downloaded: {filename}")
		except Exception as e:
			print(f"Failed to download {url}: {e}")

def main():
	parser = argparse.ArgumentParser(description="Ingest and process data files.")
	parser.add_argument('--download', action='store_true', help='Download PDFs from manifest into data/raw')
	args = parser.parse_args()
	if args.download:
		download_manifest_pdfs()
	else:
		parser.print_help()

if __name__ == "__main__":
	main()
