import sys
sys.path.append("")

import requests
import time
import urllib.parse
from bs4 import BeautifulSoup
from typing import List, Dict, Optional, Literal
from langchain_community.document_loaders import NewsURLLoader
from unstructured.cleaners.core import clean_extra_whitespace
from src.Utils.utils import GoogleTranslator

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


class NewsScraper:
    '''
    Scrape News from https://vnexpress.net/ 
    '''
    def __init__(self):
        pass

    def search_query_news(self, query: str, date_format: Literal['day', 'week', 'month', 'year'] = 'day') -> list:
        query = query.replace(' ', '%20')
        url = f"https://timkiem.vnexpress.net/?search_f=&q={query}&media_type=text&date_format={date_format}&"
        response = requests.get(url)
        attemp = 0
        max_attempts = 3
        news_items = []

        while attemp <= max_attempts:
            try:
                soup = BeautifulSoup(response.text, 'html.parser')
                articles = soup.find_all('article', class_='item-news-common')
                for article in articles:
                    url = article.get('data-url')
                    if url and 'eclick' not in url:
                        title = article.find('h3', class_='title-news').find('a').text.strip()
                        description = article.find('p', class_='description').text.strip()
                        content = self.take_text_from_link(url)
                        news_items.append({
                            'url': url,
                            'title': title,
                            'description': description,
                            'content': content
                        })
                break
            except Exception as e:
                print(f"An error occurred: {str(e)}")
                print("Resetting the Scraper in 10 seconds...")
                time.sleep(10)
                attemp += 1
                if attemp > max_attempts:
                    print("Max attempts reached. Exiting.")
                    break
        return news_items

    def take_text_from_link(self, url: str) -> str:
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            content_div = soup.find('article', class_='fck_detail')
            if content_div:
                return ' '.join([p.text.strip() for p in content_div.find_all('p')])
            return ""
        except Exception as e:
            print(f"Error fetching content from {url}: {str(e)}")
            return ""

class GoodreadsScraper:
    def __init__(self):
        self.headers = {'User-Agent': 'Mozilla/5.0'}
        self.translator = GoogleTranslator()

    def search(self, query):
        url = f"https://www.goodreads.com/search?q={query.replace(' ', '+')}"
        response = requests.get(url, headers=self.headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Get the first book link
        book = soup.select_one('tr[itemscope] a.bookTitle')
        return 'https://www.goodreads.com' + book['href'] if book else None

    def crawl_book(self, book_name):
        book_url = self.search(book_name)
        if not book_url:
            return {"error": "Book not found"}
        
        response = requests.get(book_url, headers=self.headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        data = {}
        data['title'] = soup.select_one('h1.Text__title1').text.strip()
        data['author'] = soup.select_one('.ContributorLink__name').text.strip()
        data['rating'] = soup.select_one('.RatingStatistics__rating').text.strip()
        data['description'] = soup.select_one('.DetailsLayoutRightParagraph__widthConstrained').text.strip() if soup.select_one('.DetailsLayoutRightParagraph__widthConstrained') else ''
        
        reviews = []
        for review in soup.select('.TruncatedContent__text--large')[:3]:
            review_text = review.select_one('.Formatted').text.strip()
            translated_review = self.translator.translate(review_text, 'vi')
            reviews.append(translated_review)
        data['top_reviews'] = reviews
        
        return data

    
if __name__ == "__main__":

    scraper = GoodreadsScraper()
    book_name = "Đắc Nhân Tâm"
    book_data = scraper.crawl_book(book_name)
    if "error" not in book_data:
        print(f"Book Details for {book_name}:")
        print(f"Title: {book_data['title']}")
        print(f"Author: {book_data['author']}")
        print(f"Rating: {book_data['rating']}")
        print(f"Description: {book_data['description'][:100]}...")
        print("\nTop Reviews (translated to Vietnamese):")
        for i, review in enumerate(book_data['top_reviews'], 1):
            print(f"Review {i}: {review[:100]}...")
    else:
        print(book_data["error"])
        
    client = SpiderPostClient()
    posts = client.search_posts("Đắc Nhân Tâm", page=1)
    print("posts:", posts[0])
    content = client.fetch_post_content("THE-NAO-LA-DAC-NHAN-TAM-fbc")
    print("content", content[:300], "...\n")

    scraper = NewsScraper()
    results = scraper.search_query_news(query="đắc nhân tâm", date_format="all")
    print('results',results[:2])

    