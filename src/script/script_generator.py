import sys
sys.path.append("")

import os
from typing import Dict, Any
from openai import OpenAI
import json
from src.Utils.utils import read_config
from dotenv import load_dotenv

load_dotenv()

class ScriptGenerator:
    """Generates TikTok video scripts inspired by books using OpenAI's API."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize ScriptGenerator with configuration.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary from config.yaml
        """
        self.config = config['script_generator']
        self.script_prompt_path = config.get('script_generation_prompt_path', 'config/script_generation_prompt.txt')
        self.model_name = config.get('model_name', 'gpt-4o-mini')
        self.video_time_length = config.get('video_time_length', 60)
        self.words_per_minute = config.get('words_per_minute', 140)
        self.number_of_words = int(self.video_time_length * (self.words_per_minute / 60))
        
        self.api_key = os.getenv('OPENAI_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_KEY environment variable not set")
        self.client = OpenAI(api_key=self.api_key)
        
        try:
            with open(self.script_prompt_path, 'r', encoding='utf-8') as file:
                self.prompt = file.read().format(
                    video_time_length=self.video_time_length,
                    number_of_words=self.number_of_words
                )
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt file not found at {self.script_prompt_path}")

    def generate_script(self, topic: str) -> str:
        """Generate a script for the given topic using OpenAI API.
        
        Args:
            topic (str): Topic or book title for script generation
            
        Returns:
            str: Generated script
            
        Raises:
            ValueError: If API response is invalid or JSON parsing fails
            Exception: For other API-related errors
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.prompt},
                    {"role": "user", "content": topic}
                ]
            )
            content = response.choices[0].message.content.strip()

            try:
                if content.lower().startswith("script:"):
                    script_text = content[len("script:"):].strip()
                    script = json.loads(json.dumps({"script": script_text}))["script"]
                else:
                    raise ValueError("Response does not start with 'script:'")
            except Exception as e:
                raise ValueError(f"Failed to parse script from response: {e}")

            return script
            
        except Exception as e:
            raise Exception(f"Failed to generate script: {str(e)}")

if __name__ == "__main__":
    try:
        config = read_config(path='config/config.yaml')
        generator = ScriptGenerator(config)
        test_topic = "Đắc nhân tâm"
        script = generator.generate_script(test_topic)
        print(f"Generated script for '{test_topic}':\n{script}")
    except Exception as e:
        print(f"Error: {str(e)}")