import sys
sys.path.append("")

import os
from typing import List, Tuple, Dict, Optional
import requests
from src.Utils.utils import read_config
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
            self.per_page = self.config['per_page']
            self.target_duration = self.config['target_duration']
            self.video_width = self.config['video_width']
            self.video_height = self.config['video_height']
            self.user_agent = self.config['user_agent']
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
    def get_best_video(self, query: str, used_vids: List[str]) -> Optional[str]:
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
        self.api_url = self.config['url']

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

    def get_best_video(self, query: str, used_vids: List[str] = None) -> Optional[str]:
        used_vids = used_vids or []
        try:
            vids = self.search_videos(query)
            videos = vids.get('videos', [])
            aspect_ratio = self.video_width / self.video_height
            
            filtered_videos = [
                video for video in videos 
                if (video['width'] >= self.video_width and 
                    video['height'] >= self.video_height and 
                    abs(video['width']/video['height'] - aspect_ratio) < 0.01)
            ]
            
            sorted_videos = sorted(
                filtered_videos, 
                key=lambda x: abs(self.target_duration - int(x['duration']))
            )
            
            for video in sorted_videos:
                for video_file in video['video_files']:
                    if (video_file['width'] == self.video_width and 
                        video_file['height'] == self.video_height and 
                        video_file['link'].split('.hd')[0] not in used_vids):
                        return video_file['link']
            
            print(f"No suitable videos found for query: {query} on Pexels")
            return None
        except Exception as e:
            print(f"Error processing Pexels videos for query {query}: {e}")
            return None

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

    def get_best_video(self, query: str, used_vids: List[str] = None) -> Optional[str]:
        used_vids = used_vids or []
        try:
            vids = self.search_videos(query)
            videos = vids.get('hits', [])
            aspect_ratio = self.video_width / self.video_height
            
            filtered_videos = [
                video for video in videos 
                if (video['videos']['medium']['width'] >= self.video_width and 
                    video['videos']['medium']['height'] >= self.video_height and 
                    abs(video['videos']['medium']['width']/video['videos']['medium']['height'] - aspect_ratio) < 0.01)
            ]
            
            sorted_videos = sorted(
                filtered_videos, 
                key=lambda x: abs(self.target_duration - int(x['duration']))
            )
            
            for video in sorted_videos:
                video_url = video['videos']['medium']['url']
                if video_url and video_url.split('.mp4')[0] not in used_vids:
                    return video_url
            
            print(f"No suitable videos found for query: {query} on Pixabay")
            return None
        except Exception as e:
            print(f"Error processing Pixabay videos for query {query}: {e}")
            return None

class VideoSearch:
    """Unified video search interface using Pexels as default, falling back to Pixabay."""
    
    def __init__(self, config: Dict):
        self.pexels = PexelsVideoSearch(config)
        self.pixabay = PixabayVideoSearch(config)

    def generate_video_urls(self, timed_video_searches: List[Tuple[float, float, List[str]]]) -> List[Tuple[List[float], Optional[str]]]:
        """Generate video URLs for given time ranges and search terms using Pexels first, then Pixabay.
        
        Args:
            timed_video_searches (List[Tuple[float, float, List[str]]]): List of tuples with time range and search terms.
        
        Returns:
            List[Tuple[List[float], Optional[str]]]: List of time ranges and video URLs.
        """
        timed_video_urls = []
        used_links = []
            
        for t1, t2, search_terms in timed_video_searches:
            url = None
            for query in search_terms:
                # Try Pexels first
                url = self.pexels.get_best_video(query, used_vids=used_links)
                if url:
                    used_links.append(url.split('.hd')[0])
                    break
                # Fallback to Pixabay
                url = self.pixabay.get_best_video(query, used_vids=used_links)
                if url:
                    used_links.append(url.split('.mp4')[0])
                    break
            timed_video_urls.append([[t1, t2], url])
        
        return timed_video_urls

if __name__ == "__main__":
    config = read_config(path='config/config.yaml')
    
    # Test unified VideoSearch
    video_search = VideoSearch(config)
    search_terms = [
        (0, 4.28, ['impactful book', 'communication change', 'interpersonal interaction']),
        (4.28, 14.8, ['"How to Win Friends"', 'Dale Carnegie', 'famous author']),
        (14.8, 17.34, ['true story', 'powerful principles', 'ordinary man']),
    ]
    
    print("=== Testing Unified VideoSearch (Pexels with Pixabay fallback) ===")
    video_urls = video_search.generate_video_urls(search_terms)
    for item in video_urls:
        print(item)
    
    # Test PexelsVideoSearch directly
    pexels_search = PexelsVideoSearch(config)
    print("\n=== Testing PexelsVideoSearch ===")
    pexels_url = pexels_search.get_best_video('impactful book', used_vids=[])
    print(f"Pexels video URL: {pexels_url}")
    
    # Test PixabayVideoSearch directly
    pixabay_search = PixabayVideoSearch(config)
    print("\n=== Testing PixabayVideoSearch ===")
    pixabay_url = pixabay_search.get_best_video('impactful book', used_vids=[])
    print(f"Pixabay video URL: {pixabay_url}")
            
