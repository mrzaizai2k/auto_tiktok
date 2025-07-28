import sys
sys.path.append("")

import requests
import urllib.parse
from bs4 import BeautifulSoup
from typing import List, Dict, Optional, Literal
from langchain_community.document_loaders import NewsURLLoader
from unstructured.cleaners.core import clean_extra_whitespace


class SpiderPostClient:
    BASE_SEARCH_URL = "https://spiderum.com/api/v1/search"
    BASE_POST_URL = "https://spiderum.com/bai-dang/"

    def __init__(self, user_agent: str = "Mozilla/5.0") -> None:
        self.headers = {"User-Agent": user_agent}

    def _create_search_url(self, search_text: str, page: int = 1, type_: str = "post") -> str:
        encoded_text = urllib.parse.quote(search_text)
        return f"{self.BASE_SEARCH_URL}?q={encoded_text}&page={page}&type={type_}"

    def _extract_with_bs(self, url: str) -> str:
        """Primary extraction using BeautifulSoup."""
        try:
            resp = requests.get(url, headers=self.headers, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            blocks = []
            for div in soup.select("div.ce-paragraph, div[style*='text-align']"):
                text = div.get_text(strip=True)
                if text:
                    blocks.append(text)

            return "\n\n".join(blocks).strip()
        except Exception:
            return ""

    def _extract_with_newsurl(self, url: str) -> str:
        """Fallback extraction using NewsURLLoader."""
        if not NewsURLLoader:
            return ""
        try:
            loader = NewsURLLoader(urls=[url], post_processors=[clean_extra_whitespace])
            docs = loader.load()
            return "\n\n".join(doc.page_content for doc in docs if hasattr(doc, "page_content"))
        except Exception:
            return ""

    def fetch_post_content(self, slug: str) -> str:
        """Try extracting with BeautifulSoup, fall back to NewsURLLoader if empty."""
        url = f"{self.BASE_POST_URL}{slug}"
        content = self._extract_with_bs(url)
        if not content:
            content = self._extract_with_newsurl(url)
        return content

    def search_posts(self, search_text: str, page: int = 1) -> List[Dict]:
        """Search and return posts, including full content."""
        url = self._create_search_url(search_text, page)
        response = requests.get(url, headers=self.headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("items", []):
            slug = item.get("slug")
            tags = [tag.get("name") for tag in item.get("tags", [])]
            content = self.fetch_post_content(slug) if slug else ""

            results.append({
                "title": item.get("title"),
                "link": f"{self.BASE_POST_URL}{slug}",
                "slug": slug,
                "category_slug": item.get("cat_id", {}).get("slug"),
                "description": item.get("description"),
                "created_at": item.get("created_at"),
                "tags": tags,
                "comment_count": item.get("comment_count") or 0,
                "views_count": item.get("views_count") or 0,
                "point": item.get("point") or 0,
                "content": content
            })
        return results

    def sort_posts(self, posts: List[Dict], key: Literal["comment_count", "views_count", "point"] = "views_count", 
                   reverse: bool = True) -> List[Dict]:
        """Sort posts by comment_count, views_count, or point."""

        def _to_number(v: Optional[int], default: float = 0) -> float:
            try:
                return float(v) if v is not None else default
            except (TypeError, ValueError):
                return default

        return sorted(posts, key=lambda x: _to_number(x.get(key)), reverse=reverse)



if __name__ == "__main__":
    client = SpiderPostClient()
    posts = client.search_posts("Đắc Nhân Tâm", page=1)
    print("posts:", posts[0])
    content = client.fetch_post_content("THE-NAO-LA-DAC-NHAN-TAM-fbc")
    print("content", content[:300], "...\n")