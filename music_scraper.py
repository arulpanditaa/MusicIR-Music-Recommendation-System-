import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
import urllib.parse

HEADERS = {
    "User-Agent": "JHU_601_466_MusicIR_Bot/1.0"
}
CRAWL_DELAY = 1.5 
TIMEOUT     = 10 
MIN_BIO_LEN = 100 



#keywords
MUSIC_KEYWORDS = {
    "singer", "rapper", "musician", "songwriter", "producer", "band",
    "artist", "vocalist", "dj", "duo", "group", "recording", "album",
    "genre", "hip hop", "hip-hop", "r&b", "pop", "rock", "reggaeton",
    "latin", "electronic", "country", "jazz", "soul", "funk", "punk",
    "debut", "single", "discography", "tour", "record label", "music"
}

def is_valid_music_bio(bio: str, artist_name: str) -> bool:
    """
    returns true only if the bio is about a music artist
    """
    bio_lower = bio.lower()

    #at least one keyword present
    has_music_keyword = any(kw in bio_lower for kw in MUSIC_KEYWORDS)

    #reject obvious disambiguation / wrong-topic pages
    bad_patterns = [
        r"may refer to",
        r"is a type of",
        r"is the time",      
        r"is a series of",
        r"is a record chart",    
        r"is a leave of absence",
        r"is a loud",  
        r"are social groups", 
        r"is a given name",
        r"is a japanese",
        r"is a (monthly|weekly|daily)",
    ]
    is_wrong_topic = any(re.search(p, bio_lower) for p in bad_patterns)

    return has_music_keyword and not is_wrong_topic

#wikipedia

def scrape_wikipedia(artist_name: str) -> tuple[str, str]:
    """
    fetches the first 3 paragraphs of an artist on wikipedia
    """
    encoded = urllib.parse.quote(artist_name.replace(" ", "_"))
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"

    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            bio = data.get("extract", "").strip()
            page_url = data.get("content_urls", {}).get("desktop", {}).get("page", url)
            if bio and len(bio) >= MIN_BIO_LEN and is_valid_music_bio(bio, artist_name):
                return bio, page_url

        #fallback - try musician disambiguation
        search_url = (
            f"https://en.wikipedia.org/api/rest_v1/page/summary/"
            f"{encoded}_(musician)"
        )
        resp2 = requests.get(search_url, headers=HEADERS, timeout=TIMEOUT)
        if resp2.status_code == 200:
            data2 = resp2.json()
            bio = data2.get("extract", "").strip()
            page_url = data2.get("content_urls", {}).get("desktop", {}).get("page", search_url)
            if bio and len(bio) >= MIN_BIO_LEN and is_valid_music_bio(bio, artist_name):
                return bio, page_url

        #fallback - try band disambiguation
        band_url = (
            f"https://en.wikipedia.org/api/rest_v1/page/summary/"
            f"{encoded}_(band)"
        )
        resp3 = requests.get(band_url, headers=HEADERS, timeout=TIMEOUT)
        if resp3.status_code == 200:
            data3 = resp3.json()
            bio = data3.get("extract", "").strip()
            page_url = data3.get("content_urls", {}).get("desktop", {}).get("page", band_url)
            if bio and len(bio) >= MIN_BIO_LEN and is_valid_music_bio(bio, artist_name):
                return bio, page_url

    except Exception as e:
        print(f"  [Wikipedia error] {artist_name}: {e}")

    return "", ""


#last fm

def scrape_lastfm(artist_name: str) -> tuple[str, str]:
    """
    fetches artist bio from lastfm
    """
    slug = artist_name.replace(" ", "+")
    url  = f"https://www.last.fm/music/{urllib.parse.quote(artist_name.replace(' ', '+'))}/+wiki"

    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        if resp.status_code != 200:
            return "", ""

        soup = BeautifulSoup(resp.text, "html.parser")

        bio_div = (
            soup.find("div", class_="wiki-content")
            or soup.find("div", class_="artist-wiki-content")
            or soup.find("div", {"itemprop": "description"})
        )

        if bio_div:
            #remove lastfm link text
            for a in bio_div.find_all("a"):
                a.decompose()
            bio = bio_div.get_text(separator=" ", strip=True)
            bio = re.sub(r"\s+", " ", bio).strip()
            if len(bio) >= MIN_BIO_LEN:
                return bio, url

    except Exception as e:
        print(f"  [Last.fm error] {artist_name}: {e}")

    return "", ""


#scraper class

class MusicIRScraper:

    def __init__(self):
        self.results = []

    def fetch_artist_bio(self, artist_name: str) -> dict:
        """
        tries wikipedia and lastfm as second option
        """
        print(f"Scraping: {artist_name}")

        #wikipedia
        bio, url = scrape_wikipedia(artist_name)
        if bio:
            print(f"  ✓ Wikipedia ({len(bio)} chars)")
            return {"artist": artist_name, "bio": bio, "source": "wikipedia", "url": url}
        time.sleep(CRAWL_DELAY)

        #lastfm
        bio, url = scrape_lastfm(artist_name)
        if bio and is_valid_music_bio(bio, artist_name):
            print(f"  ✓ Last.fm ({len(bio)} chars)")
            return {"artist": artist_name, "bio": bio, "source": "lastfm", "url": url}
        elif bio:
            print(f"  ✗ Last.fm bio failed music validation — discarding")
        time.sleep(CRAWL_DELAY)

        print(f"  ✗ No valid music bio found")
        return {"artist": artist_name, "bio": "", "source": "none", "url": ""}

    def scrape_all(self, artist_list: list[str]) -> pd.DataFrame:
        """
        Scrapes bios for a list of artist names.
        Enforces crawl delay between every request.
        """
        for artist in artist_list:
            result = self.fetch_artist_bio(artist)
            self.results.append(result)
            time.sleep(CRAWL_DELAY)

        return pd.DataFrame(self.results)

    def save(self, df: pd.DataFrame, path: str = "artist_metadata.csv"):
        df.to_csv(path, index=False)
        found = (df["bio"] != "").sum()
        print(f"\nSaved {len(df)} artists to {path}  ({found} with bios, {len(df)-found} empty)")


#text preprocessing=

def preprocess_bio(text: str) -> str:
    """
    clean scraped bio text for tfidf vec
    """
    text = re.sub(r"<.*?>", " ", text)#strip html
    text = re.sub(r"[^a-zA-Z\s]", " ", text)#keep letters
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def run(songs_path="dataset.csv", top_n=200,
        out_raw="artist_metadata.csv",
        out_clean="artist_metadata_clean.csv"):
    """
    Loads dataset picks top N artists by popularity scrapes bios, validates them
    """
    #top N artist
    df_songs = pd.read_csv(songs_path)

    artist_popularity = (
        df_songs.groupby("artists")["popularity"]
        .mean()
        .reset_index()
        .sort_values("popularity", ascending=False)
    )

    raw_artists = artist_popularity.head(top_n)["artists"].tolist()

    #split songs  with several artists
    artists = list(dict.fromkeys(
        a.split(";")[0].strip() for a in raw_artists
    ))

    print(f"Scraping {len(artists)} unique primary artists (from top {top_n} entries)...")
    print(f"Top 5 preview: {artists[:5]}\n")

    scraper = MusicIRScraper()
    df = scraper.scrape_all(artists)
    scraper.save(df, out_raw)

    df["bio_clean"] = df["bio"].apply(preprocess_bio)
    df.to_csv(out_clean, index=False)

    print("\nPreview:")
    print(df[["artist", "source", "bio"]].to_string(max_colwidth=80))
    return df

if __name__ == "__main__":
    run()
