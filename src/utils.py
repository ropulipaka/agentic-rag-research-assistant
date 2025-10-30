"""
Utility functions for web scraping and text processing.
"""

import re
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
from urllib.parse import urlparse
import html2text

from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def scrape_url(url: str, timeout: int = 10) -> Optional[Dict[str, str]]:
    """
    Scrape content from a URL.

    Args:
        url: URL to scrape
        timeout: Request timeout in seconds

    Returns:
        Dict with 'url', 'title', 'content', or None if failed
    """
    try:
        # Validate URL
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            print(f"⚠️  Invalid URL: {url}")
            return None

        # Fetch page
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()

        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract title
        title = soup.find('title')
        title_text = title.get_text().strip() if title else url

        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            element.decompose()

        # Convert to markdown
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = True
        content = h.handle(str(soup))

        # Clean content
        content = clean_text(content)

        return {
            'url': url,
            'title': title_text,
            'content': content
        }

    except requests.RequestException as e:
        print(f"⚠️  Failed to scrape {url}: {e}")
        return None
    except Exception as e:
        print(f"⚠️  Error processing {url}: {e}")
        return None


def clean_text(text: str) -> str:
    """
    Clean and normalize text.

    Args:
        text: Raw text

    Returns:
        Cleaned text
    """
    # Remove multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove multiple spaces
    text = re.sub(r' {2,}', ' ', text)

    # Remove leading/trailing whitespace from lines
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)

    # Remove empty lines at start/end
    text = text.strip()

    return text


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> List[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: Text to chunk
        chunk_size: Maximum chunk size in characters
        chunk_overlap: Number of characters to overlap between chunks

    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        # Get chunk
        end = start + chunk_size
        chunk = text[start:end]

        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence endings
            last_period = chunk.rfind('.')
            last_question = chunk.rfind('?')
            last_exclamation = chunk.rfind('!')

            # Use the last sentence boundary found
            last_boundary = max(last_period, last_question, last_exclamation)

            if last_boundary > chunk_size * 0.5:  # Only if boundary is not too early
                chunk = text[start:start + last_boundary + 1]
                end = start + last_boundary + 1

        chunks.append(chunk.strip())

        # Move start position with overlap
        start = end - chunk_overlap

    return chunks


def scrape_urls(urls: List[str], max_workers: int = 3) -> List[Dict[str, str]]:
    """
    Scrape multiple URLs concurrently.

    Args:
        urls: List of URLs to scrape
        max_workers: Maximum concurrent requests

    Returns:
        List of scraped documents
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    documents = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all scraping tasks
        future_to_url = {executor.submit(scrape_url, url): url for url in urls}

        # Collect results as they complete
        for future in as_completed(future_to_url):
            result = future.result()
            if result:
                documents.append(result)
                print(f"✅ Scraped: {result['title']}")

    return documents


def test_utils():
    """
    Test utility functions.
    """
    print("\n" + "="*50)
    print("Testing Utility Functions")
    print("="*50 + "\n")

    # Test text cleaning
    print("1. Testing text cleaning...")
    dirty_text = "Hello    World\n\n\n\nMultiple   spaces"
    clean = clean_text(dirty_text)
    print(f"   Input: {repr(dirty_text)}")
    print(f"   Output: {repr(clean)}")

    # Test chunking
    print("\n2. Testing text chunking...")
    long_text = "This is sentence one. " * 100  # ~2000 chars
    chunks = chunk_text(long_text, chunk_size=500, chunk_overlap=50)
    print(f"   Text length: {len(long_text)} chars")
    print(f"   Number of chunks: {len(chunks)}")
    print(f"   First chunk: {chunks[0][:100]}...")

    # Test web scraping (optional - requires internet)
    print("\n3. Testing web scraping...")
    print("   Scraping example URL...")
    result = scrape_url("https://example.com")
    if result:
        print(f"   ✅ Title: {result['title']}")
        print(f"   ✅ Content length: {len(result['content'])} chars")
        print(f"   ✅ Preview: {result['content'][:200]}...")

    print("\n" + "="*50)
    print("Test complete! ✅")
    print("="*50 + "\n")


if __name__ == "__main__":
    test_utils()