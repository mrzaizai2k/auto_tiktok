import sys
sys.path.append("")


import os
from typing import Dict
from openai import OpenAI
from dotenv import load_dotenv



class ScriptGenerator:
    """Generates TikTok video scripts inspired by books using OpenAI's API."""
    
    def __init__(self, config: Dict[str, dict]):
        """Initialize with configuration from config.yaml."""
        self.config = config['script_generator']
        self.model_name = self.config.get('model_name', 'gpt-4o-mini')
        self.model_search_name = self.config.get('model_search_name', 'gpt-4.1-mini')
        self.video_time_length = self.config.get('video_time_length', 60)
        self.words_per_minute = self.config.get('words_per_minute', 140)
        self.number_of_words = int(self.video_time_length * self.words_per_minute / 60 * 1.2)
        self.prompt_paths = {
            'web_search': self.config.get('script_generation_prompt_path', 'config/script_web_search_prompt.txt'),
            'detailed': self.config.get('script_generation_prompt_path', 'config/script_generation_detailed_prompt.txt'),
            'final': self.config.get('script_generation_prompt_path', 'config/script_generation_final_prompt.txt')
        }
        load_dotenv()
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set")
        self.client = OpenAI(api_key=self.api_key)
        self.messages = []

    def read_prompt(self, path: str) -> str:
        """Read prompt file."""
        try:
            with open(path, 'r', encoding='utf-8') as file:
                return file.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt file not found: {path}")

    def search_web(self, topic: str = "Đắc Nhân Tâm") -> str:
        """Search web for relevant information."""
        prompt = self.read_prompt(self.prompt_paths['web_search'])
        
        response = self.client.responses.create(
            model=self.model_search_name,
            tools=[{
                "type": "web_search_preview",
                    "search_context_size": self.config.get('search_context_size', "medium"),
                    }],
            input = prompt + " " + topic,
        )

        output = response.output_text

        return output

    def generate_text(self, messages: list) -> str:
        """Generate text from LLM."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise Exception(f"Failed to generate text: {e}")

    def generate_script(self, topic: str) -> str:
        """Generate script for given topic."""
        self.messages.append({"role": "assistant", "content": self.search_web(topic)})
        self.messages.append({"role": "user", "content": self.read_prompt(self.prompt_paths['detailed'])})
        detailed_script = self.generate_text(self.messages)
        self.messages.append({"role": "assistant", "content": detailed_script})
        self.messages.append({"role": "user", "content": self.read_prompt(self.prompt_paths['final'])})
        sub_final_script = self.generate_text(self.messages)
        final_script = self.extract_script(sub_final_script)
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
    script = generator.generate_script("sách Đắc Nhân Tâm")
    print(f"Generated script:\n{script}")