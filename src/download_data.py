
from pathlib import Path
from urllib.parse import urlparse
import time
import requests

from discover_links import discover_links

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

def download_datasets():
    links = discover_links()

    session = requests.Session()
    session.headers.update({
        "User-Agent": "PaddyResearchProject/1.0"
    })

    for index, (name, url) in enumerate(links.items(), 1):
        print(f"Downloading {index}: {name}")

        try:
            response = session.get(url, timeout=60)
            response.raise_for_status()

            content_type = response.headers.get(
                "Content-Type", ""
            ).lower()

            content = response.content

            if content.startswith(b"%PDF"):
                extension = ".pdf"
            elif content.startswith(b"PK\x03\x04"):
                extension = ".xlsx"
            elif content.startswith(b"\xd0\xcf\x11\xe0"):
                extension = ".xls"
            elif "text/html" in content_type:
                extension = ".html"
            else:
                extension = ".bin"

            # Stable filename based on the URL path
            filename = urlparse(url).path.rstrip("/").split("/")[-1]
            filename = f"{filename}{extension}"

            destination = RAW_DIR / filename
            destination.write_bytes(content)

            print(f"Saved: {destination}")

        except requests.RequestException as error:
            print(f"Failed: {name} - {error}")

        time.sleep(2)


if __name__ == "__main__":
    download_datasets()

