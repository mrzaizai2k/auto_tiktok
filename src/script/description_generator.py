import sys
sys.path.append("")

import os
from typing import Dict, Any
from openai import OpenAI
from src.Utils.utils import read_txt_file

from dotenv import load_dotenv

load_dotenv()


class DescriptionGenerator:
    """Generates short video descriptions from scripts using OpenAI's API."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize DescriptionGenerator with configuration.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary from config.yaml
        """
        self.config = config['description_generator']
        self.description_prompt_path = self.config.get('description_prompt_path', 'config/video_description_prompt.txt')
        self.output_description_path = self.config.get('output_description_path')
        self.model_name = self.config.get('model_name', 'gpt-4o-mini')
        
        self.api_key = os.getenv('OPENAI_API_KEY')

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = OpenAI(api_key=self.api_key)
        
        self.prompt = read_txt_file(path = self.description_prompt_path)

    def generate_description_from_llm(self, script: str) -> str:
        """Generate a short description from the given script using OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.prompt},
                    {"role": "user", "content": script}
                ]
            )
            description = response.choices[0].message.content.strip()
            if not description:
                raise ValueError("Empty description received from API")
            return description
            
        except Exception as e:
            raise Exception(f"Failed to generate description: {str(e)}")

    def save_description(self, description: str) -> None:
        """Save the generated description to a text file if a path is configured."""
        if not self.output_description_path:
            return
        try:
            os.makedirs(os.path.dirname(self.output_description_path), exist_ok=True)
            with open(self.output_description_path, 'w', encoding='utf-8') as file:
                file.write(description)
        except OSError as e:
            raise OSError(f"Failed to save description to {self.output_description_path}: {str(e)}")
    
    def generate_description(self, script: str) -> str:
        """Generate a description and save it if a path is configured."""
        description = self.generate_description_from_llm(script)
        self.save_description(description)
        return description


if __name__ == "__main__":
    from src.Utils.utils import read_config
    config = read_config(path='config/config.yaml')
    generator = DescriptionGenerator(config)
    
    # Read test script as plain text
    test_config = read_config(path='config/test_config.yaml')
    test_script = test_config.get('test_script')
    
    description = generator.generate_description(test_script)
    print(f"Generated description for test script:\n{description}")


