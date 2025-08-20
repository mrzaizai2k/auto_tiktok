import sys
sys.path.append("")


import os
import random
from typing import Dict
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

from src.script.crawl_data import SpiderPostClient, NewsScraper, GoodreadsScraper
from src.Utils.utils import read_txt_file

class ScriptGenerator:
    """Generates TikTok video scripts inspired by books using OpenAI's API."""
    
    def __init__(self, config: Dict[str, dict]):
        """Initialize with configuration from config.yaml."""
        self.config = config['script_generator']
        self.model_name = self.config.get('model_name', 'gpt-4o-mini')
        self.output_script_path = self.config.get('output_script_path', 'output/script.txt')
        self.video_time_length = self.config.get('video_time_length', 60)
        self.words_per_minute = self.config.get('words_per_minute', 140)
        self.number_of_words = int(self.video_time_length * self.words_per_minute / 60 * 1.2)
        self.prompt_paths = {
            'detailed': self.config.get('detailed_script_prompt_path', 'config/script_generation_detailed_prompt.txt'),
            'final': self.config.get('final_prompt_path', 'config/script_generation_final_prompt.txt'),
            'keyword': self.config.get('keyword_prompt_path', 'config/script_keyword_extraction_prompt.txt')
        }
        
        # Model configurations
        self.detailed_model_config = self.config.get('detailed_model', {})
        self.final_model_config = self.config.get('final_model', {})
        self.keyword_model_config = self.config.get('keyword_model', {})
        
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set")
        self.client = OpenAI(api_key=self.api_key)
        self.messages = []
    
    
    def save_script(self, script: str) -> None:
        """Save the generated description to a text file if a path is configured."""
        if not self.output_script_path:
            return
        try:
            os.makedirs(os.path.dirname(self.output_script_path), exist_ok=True)
            with open(self.output_script_path, 'w', encoding='utf-8') as file:
                file.write(script)
        except OSError as e:
            raise OSError(f"Failed to save description to {self.output_script_path}: {str(e)}")

    def extract_topic_info(self, topic: str) -> dict:
        """
        Extract keywords and book name from a given topic using the language model.
        Returns:
            dict: {
                "keywords": [kw1, kw2, kw3],
                "book_name": "book title"
            }
        """
        prompt_template = read_txt_file(path = self.prompt_paths['keyword'])
        full_prompt = f"{prompt_template.strip()} {topic.strip()}"

        result = self.generate_text(
            prompt=full_prompt, 
            config_key='keyword_model'
        ).lower()

        # Parse result
        lines = result.splitlines()
        keyword_line = next((line for line in lines if line.startswith("key:")), "")
        book_name_line = next((line for line in lines if line.startswith("book_name:")), "")

        keywords = keyword_line.replace("key:", "").strip().strip("<>").split(",")
        keywords = [kw.strip() for kw in keywords if kw.strip()]
        book_name = book_name_line.replace("book_name:", "").strip().strip("<>")
        
        topic_info = {
            "keywords": keywords,
            "book_name": book_name
        }
        return topic_info
    

    def search_web(self, topic_info: Dict) -> Dict:
        """Search web for relevant information."""
        output = {"goodreads": {}, "spiderpost": {}, "news": {}}
        
        # Goodreads: Randomly select 2 reviews
        scraper = GoodreadsScraper()
        book_name = topic_info.get("book_name", "")
        if book_name:
            try:
                book_data = scraper.crawl_book(book_name)
                if "error" not in book_data:
                    reviews = book_data.get("top_reviews", [])
                    if len(reviews) >= 2:
                        selected_reviews = random.sample(reviews, 2)
                        output["goodreads"]["content1"] = selected_reviews[0]
                        output["goodreads"]["content2"] = selected_reviews[1]
                    output["goodreads"]["description"] = book_data.get("description", "")
            except Exception as e:
                output["goodreads"]["error"] = f"Failed to fetch Goodreads data: {str(e)}"
        
        # SpiderPost: Select 1 post for book_name and 1 from all keywords
        client = SpiderPostClient()
        keywords = topic_info.get("keywords", []) + ([book_name] if book_name else [])
        book_posts = []
        all_spider_posts = []
        
        # Fetch posts for book_name
        if book_name:
            try:
                book_posts = client.search_posts(book_name, page=1)
            except Exception as e:
                output["spiderpost"]["error"] = f"Failed to fetch SpiderPost for {book_name}: {str(e)}"
        
        # Fetch posts for keywords
        for keyword in topic_info.get("keywords", []):
            try:
                posts = client.search_posts(keyword, page=1)
                all_spider_posts.extend(posts)
            except Exception as e:
                output["spiderpost"]["error"] = f"Failed to fetch SpiderPost for {keyword}: {str(e)}"
        
        # Select non-empty posts
        if book_posts:
            valid_book_posts = [p for p in book_posts if p.get("content")]
            if valid_book_posts:
                output["spiderpost"]["content1"] = random.choice(valid_book_posts)["content"]
        
        if all_spider_posts:
            valid_spider_posts = [p for p in all_spider_posts if p.get("content")]
            if valid_spider_posts:
                output["spiderpost"]["content2"] = random.choice(valid_spider_posts)["content"]
        
        # NewsScraper: Select 1 news for book_name and 1 from all keywords
        scraper = NewsScraper()
        book_news = []
        all_news = []
        
        # Fetch news for book_name
        if book_name:
            try:
                book_news = scraper.search_query_news(query=book_name, date_format="all")
            except Exception as e:
                output["news"]["error"] = f"Failed to fetch news for {book_name}: {str(e)}"
        
        # Fetch news for keywords
        for keyword in topic_info.get("keywords", []):
            try:
                results = scraper.search_query_news(query=keyword, date_format="all")
                all_news.extend(results)
            except Exception as e:
                output["news"]["error"] = f"Failed to fetch news for {keyword}: {str(e)}"
        
        # Select non-empty news
        if book_news:
            valid_book_news = [n for n in book_news if n.get("content")]
            if valid_book_news:
                output["news"]["content1"] = random.choice(valid_book_news)["content"]
        
        if all_news:
            valid_news = [n for n in all_news if n.get("content")]
            if valid_news:
                output["news"]["content2"] = random.choice(valid_news)["content"]
        
        return output

    def generate_reference(self, search_output: Dict) -> str:
        """Generate reference text from search_web output."""
        reference_text = []
        
        for source, data in search_output.items():
            
                
            if source == "goodreads":
                reference_text.append(f"\nĐây là những bình luận từ {source.capitalize()}:")
            else:
                reference_text.append(f"\nĐây là những bài viết từ {source.capitalize()}:")
                
            for key, content in data.items():
                if key.startswith("content"):
                    reference_text.append(f"{key.capitalize()}: {content}")

        return "\n".join(reference_text)
    
    def generate_text(self, prompt: str, config_key: str = 'default', number_of_words: int = None) -> str:
        """Generate text from LLM using configuration."""
        # Get model config based on config_key
        if config_key == 'detailed_model':
            model_config = self.detailed_model_config
        elif config_key == 'final_model':
            model_config = self.final_model_config
        elif config_key == 'keyword_model':
            model_config = self.keyword_model_config
        else:
            model_config = {}
        
        # Default values
        model_name = model_config.get('model_name', self.model_name)
        temperature = model_config.get('temperature', 0.55)
        max_tokens = model_config.get('max_tokens', 512)
        top_p = model_config.get('top_p', 0.5)
        
        # Replace {number_of_words} in prompt if number_of_words is provided
        if number_of_words is not None:
            prompt = prompt.replace('{number_of_words}', str(number_of_words))
        
        try:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise Exception(f"Failed to generate text: {e}")

    def generate_script(self, topic: str) -> str:
        """Generate script for given topic."""
        topic_info = self.extract_topic_info(topic)
        content = self.search_web(topic_info)
        reference = self.generate_reference(content)
        
        detailed_prompt = read_txt_file(self.prompt_paths['detailed']) + " " + reference
        detailed_script = self.generate_text(
            prompt=detailed_prompt, 
            config_key='detailed_model'
        )
        
        final_prompt = read_txt_file(self.prompt_paths['final']) + " " + detailed_script 
        sub_final_script = self.generate_text(
            prompt=final_prompt,
            config_key='final_model',
            number_of_words=self.number_of_words
        )
        
        final_script = self.extract_script(sub_final_script)
        self.save_script(script=final_script)
        return final_script

    def extract_script(self, content: str) -> str:
        """Extract script from response."""
        if not content.lower().startswith("script:"):
            raise ValueError("Response must start with 'script:'")
        return content[len("script:"):].strip()

if __name__ == "__main__":
    from src.Utils.utils import read_config
    config = read_config('config/config.yaml')
    generator = ScriptGenerator(config)
    script = generator.generate_script("sách Đắc nhân tâm")
    print(f"Generated script:\n{script}")