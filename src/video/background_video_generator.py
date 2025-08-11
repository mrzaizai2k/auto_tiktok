import sys
sys.path.append("")

import os
from typing import List, Tuple, Dict, Optional
import requests
from src.video.image_text_matching import ImageTextSimilarity
from src.Utils.utils import read_config, timeit, GoogleTranslator
from abc import ABC, abstractmethod

from dotenv import load_dotenv

load_dotenv()


class VideoSearchBase(ABC):
    """Abstract base class for video search implementations."""
    
    def __init__(self, config: Dict):
        """Initialize with configuration.
        
        Args:
            config (Dict): Configuration dictionary containing API settings.
        
        Raises:
            KeyError: If required configuration parameters are missing.
            ValueError: If configuration values are invalid.
        """
        try:
            self.config = config["video_search"]
            self.per_page = self.config.get('per_page', 15)
            self.video_width = self.config.get('video_width', 1920)
            self.video_height = self.config.get('video_height', 1080)
            self.user_agent = self.config.get('user_agent', 'Mozilla/5.0')
            self.ratio_threshold = self.config.get('ratio_threshold', 0.1)  # Aspect ratio threshold for filtering videos
        except KeyError as e:
            raise KeyError(f"Missing configuration key: {e}")

    @abstractmethod
    def search_videos(self, query: str) -> Dict:
        """Search videos on the API.
        
        Args:
            query (str): Search query string.
        
        Returns:
            Dict: API response JSON.
        """
        pass

    @abstractmethod
    def get_best_videos(self, query: str, used_vids: List[str]) -> Optional[str]:
        """Get best video URL matching criteria.
        
        Args:
            query (str): Search query string.
            used_vids (List[str]): List of used video URLs.
        
        Returns:
            Optional[str]: Video URL or None if no suitable video found.
        """
        pass

class PexelsVideoSearch(VideoSearchBase):
    """Pexels-specific video search implementation."""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.pexels_api_key: str = os.getenv('PEXELS_API_KEY')
        if not self.pexels_api_key:
            raise ValueError("PEXELS_API_KEY not found in environment variables")
        self.api_url = "https://api.pexels.com/videos/search"

    def search_videos(self, query: str) -> Dict:
        headers = {
            "Authorization": self.pexels_api_key,
            "User-Agent": self.user_agent
        }
        params = {
            "query": query,
            "orientation": "landscape" if self.video_width > self.video_height else "portrait",
            "per_page": self.per_page
        }
        
        try:
            response = requests.get(self.api_url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise requests.RequestException(f"Pexels API request failed: {e}")
        
    def get_best_videos(self, search_terms: list[str], 
                        used_vids: List[str] = None, 
                        duration: float = None,
                        max_videos:int = 3) -> List[Tuple[str, str]]:
        """
        Retrieves a list of video URLs and their corresponding image links from Pexels based on search terms.
        Filters videos by minimum width, height, aspect ratio, and optional duration.
        Excludes previously used videos and handles errors gracefully.

        Args:
            search_terms (list[str]): List of search queries.
            used_vids (List[str], optional): List of video URLs to exclude. Defaults to None.
            duration (float, optional): Minimum video duration in seconds. Defaults to None.

        Returns:
            List[Tuple[str, str]]: List of tuples containing (video URL, video image link).
        """
        used_vids = used_vids or []
        video_data = []
        try:
            for query in search_terms:
                vids = self.search_videos(query)
                videos = vids.get('videos', [])
                aspect_ratio = self.video_width / self.video_height
                
                filtered_videos = [
                    video for video in videos 
                    if (video['width'] >= self.video_width and 
                        video['height'] >= self.video_height and 
                        abs(video['width']/video['height'] - aspect_ratio) < self.ratio_threshold and
                        (not duration or int(video['duration']) >= duration))
                ]
                
                for video in filtered_videos[:max_videos]:
                    for video_file in video['video_files']:
                        if video_file['link'].split('.hd')[0] not in used_vids:
                            video_data.append((video_file['link'], video['image']))
                            break  # Take only one video file per video
                
            if not video_data:
                print(f"No suitable videos found for queries: {search_terms} on Pexels")
            return video_data
        except Exception as e:
            print(f"Error processing Pexels videos for queries {search_terms}: {e}")
            return []

class PixabayVideoSearch(VideoSearchBase):
    """Pixabay-specific video search implementation."""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.pixabay_api_key: str = os.getenv('PIXABAY_API_KEY')
        if not self.pixabay_api_key:
            raise ValueError("PIXABAY_API_KEY not found in environment variables")
        self.api_url = "https://pixabay.com/api/videos/"

    def search_videos(self, query: str) -> Dict:
        headers = {"User-Agent": self.user_agent}
        params = {
            "key": self.pixabay_api_key,
            "query": query,
            "min_width": self.video_width,
            "min_height": self.video_height,
            "per_page": self.per_page
        }
        
        try:
            response = requests.get(self.api_url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise requests.RequestException(f"Pixabay API request failed: {e}")

    def get_best_videos(self, search_terms: list[str], 
                        used_vids: List[str] = None, 
                        duration: float = None,
                        max_videos: int = 3) -> List[Tuple[str, str]]:
        """
        Retrieves a list of video URLs and their corresponding image links from Pixabay based on search terms.
        Filters videos by minimum width, height, aspect ratio, and optional duration.
        Excludes previously used videos and handles errors gracefully.

        Args:
            search_terms (list[str]): List of search queries.
            used_vids (List[str], optional): List of video URLs to exclude. Defaults to None.
            duration (float, optional): Minimum video duration in seconds. Defaults to None.

        Returns:
            List[Tuple[str, str]]: List of tuples containing (video URL, video image link).
        """
        used_vids = used_vids or []
        video_data = []
        try:
            for query in search_terms:
                vids = self.search_videos(query)
                videos = vids.get('hits', [])
                aspect_ratio = self.video_width / self.video_height
                
                filtered_videos = [
                    video for video in videos 
                    if (video['videos']['medium']['width'] >= self.video_width and 
                        video['videos']['medium']['height'] >= self.video_height and 
                        abs(video['videos']['medium']['width']/video['videos']['medium']['height'] - aspect_ratio) < self.ratio_threshold and
                        (not duration or video['duration'] >= duration))
                ]
                
                for video in filtered_videos[:max_videos]:
                    video_url = video['videos']['medium']['url']
                    image_url = video['videos']['medium']['thumbnail']
                    if video_url.split('.mp4')[0] not in used_vids:
                        video_data.append((video_url, image_url))
                    
            if not video_data:
                print(f"No suitable videos found for queries: {search_terms} on Pixabay")
            return video_data
        except Exception as e:
            print(f"Error processing Pixabay videos for queries {search_terms}: {e}")
            return []        
        

class VideoSearch:
    """Unified video search interface using Pexels as default, falling back to Pixabay."""
    
    def __init__(self, config: Dict):
        self.pexels = PexelsVideoSearch(config)
        self.pixabay = PixabayVideoSearch(config)
        self.image_text_similarity = ImageTextSimilarity(config)
        self.translator = GoogleTranslator()


    @timeit
    def get_best_video_url(self, video_image_pairs: List[Tuple[str, str]], sentence: str, used_links: List[str]) -> Optional[str]:
        """
        Select the best video URL based on image-text similarity.

        Args:
            video_image_pairs (List[Tuple[str, str]]): List of (video URL, image URL) tuples.
            sentence (str): Sentence for similarity comparison.
            used_links (List[str]): List of used video URLs to exclude.

        Returns:
            Optional[str]: Best video URL or None if no match.
        """
        if not video_image_pairs:
            return None
        
        image_urls = [pair[1] for pair in video_image_pairs]
        best_image_url = self.image_text_similarity.get_similarity(image_paths=image_urls, text=sentence, sort='des')[0][0]
        
        for video_url, image_url in video_image_pairs:
            if image_url == best_image_url and video_url.split('.hd')[0] not in used_links:
                used_links.append(video_url.split('.hd')[0])
                return video_url
        
        return None

    def generate_video_urls(self, timed_video_searches: List[Tuple[float, float, List[str], str]]) -> List[Tuple[List[float], Optional[str]]]:
        timed_video_urls = []
        used_links = []
            
        for t1, t2, search_terms, sentence in timed_video_searches:
            duration = abs(t2 - t1)
            sentence = self.translator.translate(text=sentence, to_lang='en')
            
            # Try Pexels
            video_image_pairs = self.pexels.get_best_videos(search_terms=search_terms, 
                                                            used_vids=used_links,
                                                            duration=duration)
            if not video_image_pairs:
                # Fallback to Pixabay if no Pexels videos found
                video_image_pairs = self.pixabay.get_best_videos(search_terms=search_terms, 
                                                                  used_vids=used_links,
                                                                  duration=duration)

            
            url = self.get_best_video_url(video_image_pairs, sentence, used_links)
            
            timed_video_urls.append([[t1, t2], url])
        
        return timed_video_urls

if __name__ == "__main__":
    config = read_config(path='config/config.yaml')
    
    # Test unified VideoSearch
    video_search = VideoSearch(config)
    search_terms = [
        (0, 4.28, ['impactful book', 'communication change', 'interpersonal interaction'], "Cuốn sách đầy cảm hứng thúc đẩy thay đổi tích cực."),
        (4.28, 74.8, ['How to Win Friends', 'Dale Carnegie', 'famous author'], "Sách của Carnegie truyền cảm hứng qua kinh nghiệm thực tiễn"),
        (74.8, 124.14, ['true story', 'powerful principles', 'ordinary man'], "Câu chuyện có thật về người bình thường áp dụng nguyên tắc mạnh mẽ."),
        (124.14, 128.32, ['false living', 'losing identity', 'inauthentic existence'], "Mất đi bản ngã vì sống không đúng với chính mình.")
    ]
    
    print("=== Testing Unified VideoSearch (Pexels with Pixabay fallback) ===")
    video_urls = video_search.generate_video_urls(timed_video_searches=search_terms)
    for item in video_urls:
        print(item)
    
    # Test PexelsVideoSearch directly
    pexels_search = PexelsVideoSearch(config)
    print("\n=== Testing PexelsVideoSearch ===")
    pexels_url = pexels_search.get_best_videos(search_terms=['impactful book'], used_vids=[])
    print(f"Pexels video URL: {pexels_url}")
    
    # Test PixabayVideoSearch directly
    pixabay_search = PixabayVideoSearch(config)
    print("\n=== Testing PixabayVideoSearch ===")
    pixabay_url = pixabay_search.get_best_videos(search_terms=['impactful book'], used_vids=[])
    print(f"Pixabay video URL: {pixabay_url}")
            
