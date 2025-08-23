import sys
sys.path.append("")

import requests
import time
import urllib.parse
from bs4 import BeautifulSoup
from typing import List, Dict, Optional, Literal
from langchain_community.document_loaders import NewsURLLoader
from unstructured.cleaners.core import clean_extra_whitespace
from src.Utils.utils import GoogleTranslator, timeit
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

class SpiderPostClient:
    BASE_SEARCH_URL = "https://spiderum.com/api/v1/search"
    BASE_POST_URL = "https://spiderum.com/bai-dang/"
    BASE_TOP_URL = "https://spiderum.com/api/v1/feed/getAllPosts?type=top&page="

    def __init__(self, user_agent: str = "Mozilla/5.0") -> None:
        self.headers = {"User-Agent": user_agent}

    def _create_search_url(self, search_text: str, page: int = 1, type_: str = "post") -> str:
        """Create URL for searching posts by text."""
        return f"{self.BASE_SEARCH_URL}?q={urllib.parse.quote(search_text)}&page={page}&type={type_}"

    def _create_top_url(self, page: int = 1) -> str:
        """Create URL for fetching top posts."""
        return f"{self.BASE_TOP_URL}{page}"

    def _extract_with_bs(self, url: str) -> str:
        """Extract text content from a URL using BeautifulSoup."""
        try:
            resp = requests.get(url, headers=self.headers, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            blocks = [div.get_text(strip=True) for div in soup.select("div.ce-paragraph, div[style*='text-align']") if div.get_text(strip=True)]
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
        """Fetch and extract content for a post by slug."""
        url = f"{self.BASE_POST_URL}{slug}"
        content = self._extract_with_bs(url)
        return content if content else self._extract_with_newsurl(url)

    def process_results(self, items: List[Dict], base_url: str) -> List[Dict]:
        """Process API response items into standardized format and fetch content."""
        results, slugs = [], []
        for item in items:
            slug = item.get("slug")
            results.append({
                "title": item.get("title"),
                "link": f"{base_url}{slug}",
                "slug": slug,
                "category_slug": item.get("cat_id", {}).get("slug", ""),
                "description": item.get("description"),
                "created_at": item.get("created_at"),
                "tags": [tag.get("name") for tag in item.get("tags", [])],
                "comment_count": item.get("comment_count", 0),
                "views_count": item.get("views_count", 0),
                "point": item.get("point", 0),
                "content": ""
            })
            if slug:
                slugs.append(slug)

        with ThreadPoolExecutor() as executor:
            contents = list(executor.map(self.fetch_post_content, slugs))
        for result, content in zip(results, contents):
            result["content"] = content if isinstance(content, str) else ""
        return results

    @timeit
    def search_posts(self, search_text: str = "", page: int = 1) -> List[Dict]:
        """Search posts by topic or fetch top posts if no search text provided."""
        try:
            return self.fetch_by_topic(search_text, page) if search_text else self.fetch_top_posts(page)
        except Exception as e:
            print(f"Search failed: {e}")
            return []

    @timeit
    def fetch_by_topic(self, search_text: str, page: int = 1) -> List[Dict]:
        """Fetch posts by search topic."""
        try:
            url = self._create_search_url(search_text, page)
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            return self.process_results(data.get("items", []), self.BASE_POST_URL)
        except Exception as e:
            print(f"Fetch by topic failed: {e}")
            return []

    @timeit
    def fetch_top_posts(self, page: int = 1) -> List[Dict]:
        """Fetch top posts from the API."""
        try:
            url = self._create_top_url(page)
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            return self.process_results(data["posts"]["items"], self.BASE_POST_URL)
        except Exception as e:
            print(f"Fetch top posts failed: {e}")
            return []
            

class NewsScraper:
    '''
    Scrape News from https://vnexpress.net/ 
    '''
    def __init__(self):
        pass

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

    @timeit
    def search_query_news(self, query: str, date_format: Literal['day', 'week', 'month', 'year'] = 'day') -> list:
        query = query.replace(' ', '%20')
        url = f"https://timkiem.vnexpress.net/?search_f=&q={query}&media_type=text&date_format={date_format}&"
        news_items = []
        urls = []

        # Fetch and parse search page
        attemp = 0
        max_attempts = 3
        while attemp <= max_attempts:
            try:
                response = requests.get(url)
                soup = BeautifulSoup(response.text, 'html.parser')
                articles = soup.find_all('article', class_='item-news-common')
                for article in articles:
                    article_url = article.get('data-url')
                    if article_url and 'eclick' not in article_url:
                        title = article.find('h3', class_='title-news').find('a').text.strip()
                        description = article.find('p', class_='description').text.strip()
                        news_items.append({
                            'url': article_url,
                            'title': title,
                            'description': description,
                            'content': ""
                        })
                        urls.append(article_url)
                break
            except Exception as e:
                print(f"An error occurred: {str(e)}")
                print("Resetting the Scraper in 10 seconds...")
                time.sleep(10)
                attemp += 1
                if attemp > max_attempts:
                    print("Max attempts reached. Exiting.")
                    return news_items

        # Fetch content in parallel using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor() as executor:
            contents = list(executor.map(self.take_text_from_link, urls))

        # Assign content to news_items
        for item, content in zip(news_items, contents):
            item['content'] = content if isinstance(content, str) else ""

        return news_items

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

    book_name = "Đắc Nhân Tâm"
        
    client = SpiderPostClient()
    posts = client.search_posts(search_text="sách Đắc Nhân Tâm", page=1)

    # posts = client.search_posts(book_name, page=1)
    # print("posts:", posts[0])
    # content = client.fetch_post_content("THE-NAO-LA-DAC-NHAN-TAM-fbc")
    # print("content", content[:30], "...\n")



    # scraper = NewsScraper()
    # results = scraper.search_query_news(query=book_name, date_format="all")
    # print('results',results[0]['url'])

    # scraper = GoodreadsScraper()
    
    # book_data = scraper.crawl_book(book_name)
    # if "error" not in book_data:
    #     print(f"Book Details for {book_name}:")
    #     print(f"Title: {book_data['title']}")
    #     print(f"Author: {book_data['author']}")
    #     print(f"Rating: {book_data['rating']}")
    #     print(f"Description: {book_data['description'][:100]}...")
    #     print("\nTop Reviews (translated to Vietnamese):")
    #     for i, review in enumerate(book_data['top_reviews'], 1):
    #         print(f"Review {i}: {review[:30]}...")
    # else:
    #     print(book_data["error"])

    