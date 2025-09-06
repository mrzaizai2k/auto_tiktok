import sys
sys.path.append("")


import os
import random
from typing import Dict
from openai import OpenAI
import random
from dotenv import load_dotenv
load_dotenv()

from src.script.crawl_data import SpiderPostClient
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
        }
        self.topic = self.config.get('topic', None)
        
        # Model configurations
        self.detailed_model_config = self.config.get('detailed_model', {})
        self.final_model_config = self.config.get('final_model', {})
        
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

    def search_web(self) -> Dict:
        """Search web for relevant information from Spiderum."""
        output = {"spiderpost": {}}
        
        # SpiderPost: Select 1 post for topic
        client = SpiderPostClient()
        try:
            page  = self.config.get('page', None)
            if page is None:
                page = random.randint(1, 20)
            posts = client.search_posts(search_text=self.topic, page = page)
            # Filter posts with content >= 500 words
            valid_posts = [p for p in posts if p.get("content") and len(p["content"].split()) >= 500]
            if valid_posts:
                # Sort by views_count
                sorted_posts = client.sort_posts(valid_posts, key="views_count", reverse=True)
                output = random.choice(sorted_posts)
        except Exception as e:
            print(f"Error fetching SpiderPost data: {e}")
        
        return output

    def generate_reference(self, search_output: Dict) -> str:
        """Generate reference text from search_web output."""
        reference_text = "Đây là bài viết gốc từ Spiderum:\n"
        
        reference_text = reference_text + f"Tiêu đề: {search_output.get('title', 'N/A')}\n"
        reference_text = reference_text + f"Bài viết: {search_output.get('content', 'N/A')}\n"
        return reference_text
    
    def generate_text(self, prompt: str, config_key: str = 'default', number_of_words: int = None) -> str:
        """Generate text from LLM using configuration."""
        if config_key == 'detailed_model':
            model_config = self.detailed_model_config
        elif config_key == 'final_model':
            model_config = self.final_model_config
        else:
            model_config = {}
        
        model_name = model_config.get('model_name', self.model_name)
        temperature = model_config.get('temperature', 0.55)
        max_tokens = model_config.get('max_tokens', 512)
        top_p = model_config.get('top_p', 0.5)
        
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

    
    def generate_script(self) -> str:
        """Generate script for topic from config."""
        post = self.search_web()
        reference = self.generate_reference(search_output=post)
        
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
    script = generator.generate_script()
    print(f"Generated script:\n{script}")