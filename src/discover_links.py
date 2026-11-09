
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL = (
    "https://www.statistics.gov.lk/"
    "Agriculture/StaticalInformation/PaddyStatistics"
)

def discover_links():
    response = requests.get(BASE_URL, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    keywords = [
        "Average Yield",
        "Average Yeild",
        "Harvested Extent",
        "Sown Extent",
        "Production",
    ]

    links = {}

    for a in soup.find_all("a", href=True):
        text = a.get_text(" ", strip=True)
        href = a["href"]

        if not any(k.lower() in text.lower() for k in keywords):
            continue

        if not any(k in href.lower() for k in
                   ["1979-2025", "1979_2025"]):
            continue

        url = urljoin(BASE_URL, href)
        links[text] = url

    return links


if __name__ == "__main__":
    links = discover_links()

    for name, url in links.items():
        print(name)
        print(url)
        print()

    print("Found:", len(links), "datasets")
