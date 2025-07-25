import sys
sys.path.append("")

import os
from typing import List, Tuple, Dict, Optional
import requests
from src.Utils.utils import read_config
from dotenv import load_dotenv

load_dotenv()

class VideoSearch:
    """A class to search and retrieve video URLs from Pexels API based on configuration."""
    
    def __init__(self, config: Dict):
        """Initialize VideoSearch with configuration.
        
        Args:
            config (Dict): Configuration dictionary containing API settings and parameters.
        
        Raises:
            KeyError: If required configuration parameters are missing.
            ValueError: If configuration values are invalid.
        """
        self.pexels_key: str = os.getenv('PEXELS_KEY')
        if not self.pexels_key:
            raise ValueError("PEXELS_KEY not found in environment variables")
        
        try:
            self.config = config["video_search"]
            self.api_url = self.config['url']
            self.per_page = self.config['per_page']
            self.target_duration = self.config['target_duration']
            self.video_width = self.config['video_width']
            self.video_height = self.config['video_height']
            self.user_agent = self.config['user_agent']
        except KeyError as e:
            raise KeyError(f"Missing configuration key: {e}")

    def search_videos(self, query: str) -> Dict:
        """Search videos on Pexels API.
        
        Args:
            query (str): Search query string.
        
        Returns:
            Dict: API response JSON.
        
        Raises:
            requests.RequestException: If API request fails.
        """
        headers = {
            "Authorization": self.pexels_key,
            "User-Agent": self.user_agent
        }
        if self.video_width > self.video_height:
            orientation = "landscape"
        else:
            orientation = "portrait"

        params = {
            "query": query,
            "orientation": orientation,
            "per_page": self.per_page
        }
        
        try:
            response = requests.get(self.api_url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise requests.RequestException(f"API request failed: {e}")

    def get_best_video(self, query: str,  
                    used_vids: List[str] = None) -> Optional[str]:
        """Get best video URL matching criteria.
        
        Args:
            query (str): Search query string.
            used_vids (List[str]): List of used video URLs to avoid duplicates.
        
        Returns:
            Optional[str]: Video URL or None if no suitable video found.
        """
        used_vids = used_vids or []
        try:
            vids = self.search_videos(query)
            videos = vids.get('videos', [])
            
            # Swap width/height for portrait orientation
            aspect_ratio = self.video_width  / self.video_height
            
            filtered_videos = [
                video for video in videos 
                if (video['width'] >= self.video_width and 
                    video['height'] >= self.video_height  and 
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
            
            print(f"No suitable videos found for query: {query}")
            return None
        except Exception as e:
            print(f"Error processing videos for query {query}: {e}")
            return None

    def generate_video_urls(self, timed_video_searches: List[Tuple[float, float, List[str]]], 
                          video_server: str) -> List[Tuple[List[float], Optional[str]]]:
        """Generate video URLs for given time ranges and search terms.
        
        Args:
            timed_video_searches (List[Tuple[float, float, List[str]]]): List of tuples with time range and search terms.
            video_server (str): Video server name (e.g., 'pexel').
        
        Returns:
            List[Tuple[List[float], Optional[str]]]: List of time ranges and video URLs.
        
        Raises:
            ValueError: If unsupported video server is specified.
        """
        if video_server.lower() != "pexel":
            raise ValueError("Unsupported video server specified")
        
        timed_video_urls = []
        used_links = []
            
        for t1, t2, search_terms in timed_video_searches:
            url = None
            for query in search_terms:
                url = self.get_best_video(query, used_vids=used_links)
                if url:
                    used_links.append(url.split('.hd')[0])
                    break
            timed_video_urls.append([[t1, t2], url])
        
        return timed_video_urls


if __name__ == "__main__":
    try:
        config = read_config(path='config/config.yaml')
        video_search = VideoSearch(config)
        
        search_terms = [
            (0, 4.28, ['impactful book', 'communication change', 'interpersonal interaction']),
            (4.28, 14.8, ['"How to Win Friends"', 'Dale Carnegie', 'famous author']),
            (14.8, 17.34, ['true story', 'powerful principles', 'ordinary man']),
        ]
        
        video_urls = video_search.generate_video_urls(search_terms, "pexel")
        
        print("=== Video URLs ===")
        for item in video_urls:
            print(item)
            
    except Exception as e:
        print(f"Error: {e}")
