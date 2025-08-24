import sys
sys.path.append("")

import torch
from PIL import Image
from torch.nn import functional as F
from transformers import AutoProcessor, AutoModelForCausalLM
import requests
from io import BytesIO
from typing import List, Optional, Tuple, Literal
from src.Utils.utils import timeit, read_config, take_device

class ImageTextSimilarity:
    """Class to compute similarity scores between images and text using a pre-trained model."""

    
    def __init__(self, config):
        """Initialize with config file path.
        
        Args:
            config_path (str): Path to the configuration YAML file.
        """
        self.config = config["image_text_matching"]
        self.model= None
        self.processor= None
        self.text_prompt = "<|user|>\n<sent>\nSummary above sentence: <|end|>\n<|assistant|>\n"
        self.img_prompt = "<|user|>\n<|image_1|>\nSummary above image: <|end|>\n<|assistant|>\n"
        self.device = take_device()
        self.load_model()


    def load_model(self) -> None:
        """Load the model and processor from the configured base model path."""
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config["base_model_path"],
            device_map=self.device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=self.config.get("trust_remote_code", True)
        )
        self.processor = AutoProcessor.from_pretrained(
            self.config["base_model_path"],
            trust_remote_code=self.config.get("trust_remote_code", True),
            torch_dtype=torch.bfloat16,
        )
        self.processor.tokenizer.padding_side = self.config["padding_side"]
        self.processor.tokenizer.padding = self.config["padding"]

    def get_new_size(self, img: Image.Image, target_height: int = 480) -> tuple:
        """Calculate new image dimensions with target height, maintaining aspect ratio.
        
        Args:
            img (Image.Image): Input PIL image.
            target_height (int): Desired height (default: 720).
            
        Returns:
            tuple: (new_width, new_height)
        """
        aspect_ratio = img.width / img.height
        new_height = target_height
        new_width = int(new_height * aspect_ratio)
        return new_width, new_height

    def load_image(self, image_path: str) -> Image.Image:
        """Load and resize an image to height=720 while maintaining aspect ratio.
        
        Args:
            image_path (str): Path or URL to the image.
            
        Returns:
            Image.Image: Resized PIL image.
            
        Raises:
            Exception: If image loading fails.
        """
        if image_path.startswith('http'):
            response = requests.get(image_path)
            img = Image.open(BytesIO(response.content))
        else:
            img = Image.open(image_path)
        
        # Get new dimensions and resize
        new_width, new_height = self.get_new_size(img)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return img
    
    def _sort_key(self, item: Tuple[str, Optional[float]], sort: Literal['asc', 'des']) -> float:
        """Sort key function for image similarity results.
        
        Args:
            item (Tuple[str, Optional[float]]): Tuple of image path and similarity score.
            sort (Literal['asc', 'des']): Sort order, 'asc' for ascending, 'des' for descending.
            
        Returns:
            float: Sort key value, handling None scores.
        """
        return item[1] if item[1] is not None else float('-inf') if sort == 'des' else float('inf')

    @timeit
    def get_similarity(self, image_paths: List[str], text: str, sort: Literal['asc', 'des'] = 'des') -> List[Tuple[str, Optional[float]]]:
        """Compute similarity scores between a list of images and text.
        
        Args:
            image_paths (List[str]): List of image paths or URLs.
            text (str): Text to compare against images.
            sort (Literal['asc', 'des']): Sort order, 'asc' for ascending, 'des' for descending. Defaults to 'des'.
            
        Returns:
            List[Tuple[str, Optional[float]]]: List of tuples containing image path and similarity score, None for failed images.
        """
        # Prepare text input
        input_texts = self.text_prompt.replace('<sent>', text)
        inputs_text = self.processor(
            text=input_texts,
            images=None,
            return_tensors="pt",
            padding=self.config["padding"]
        )
        for key in inputs_text:
            inputs_text[key] = inputs_text[key].to(self.device)

        # Get text embedding
        with torch.no_grad():
            emb_text = self.model(**inputs_text, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :]
            emb_text = F.normalize(emb_text, dim=-1)

        # Process each image and compute similarity
        results = []
        for image_path in image_paths:
            try:
                input_image = [self.load_image(image_path)]
                inputs_image = self.processor(
                    text=self.img_prompt,
                    images=input_image,
                    return_tensors="pt",
                    padding=self.config["padding"]
                ).to(self.device)

                with torch.no_grad():
                    emb_image = self.model(**inputs_image, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :]
                    emb_image = F.normalize(emb_image, dim=-1)
                    score = (emb_image @ emb_text.T).item()
                    results.append((image_path, score))
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append((image_path, None))

        # Sort results using separate sort key function
        return sorted(results, key=lambda item: self._sort_key(item, sort), reverse=(sort == 'des'))


if __name__ == "__main__":
    # Example usage
    config_path = "config/config.yaml"
    config = read_config(config_path)
    matcher = ImageTextSimilarity(config=config)
    test_config = read_config("config/test_config.yaml")
    image_paths = [
        "https://cdn.pixabay.com/video/2024/08/30/228847_medium.jpg",
        "https://images.pexels.com/videos/33304325/black-amp-white-black-and-white-black-and-white-portrait-book-33304325.jpeg?auto=compress&cs=tinysrgb&fit=crop&h=1200&w=630",
        test_config['image_path'],  
        "https://images.pexels.com/videos/7608710/baccarat-bar-bets-blackjack-7608710.jpeg?auto=compress&cs=tinysrgb&fit=crop&h=1200&w=630"
    ]
    text = "Inspirational book that promotes positive change."
    results = matcher.get_similarity(image_paths, text, sort='des')
    print("Similarity Scores:", results)
    