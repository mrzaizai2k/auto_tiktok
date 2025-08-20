import sys
sys.path.append("")

import os
import re
from typing import List, Dict, Any, Tuple

from openai import OpenAI
import concurrent.futures
from dotenv import load_dotenv
from src.Utils.utils import read_config, read_txt_file

load_dotenv()

class VideoKeywordGenerator:
    """Generates visually concrete keywords for video segments based on a plain script (no timestamps)."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config['keyword_generator']
        self.model_name = self.config.get('model_name', 'gpt-4o-mini')
        self.temperature = self.config.get('temperature', 1.0)
        self.max_tokens = self.config.get('max_tokens', 4096)
        self.top_p = self.config.get('top_p', 0.8)
        self.presence_penalty = self.config.get('presence_penalty', 0.0)
        self.video_search_keyword_prompt_path = self.config.get(
            'video_search_keyword_prompt_path',
            'config/video_search_keyword_prompt.txt'
        )
        self.split_length = self.config.get('split_length', 500)
        
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = OpenAI(api_key=self.api_key)

        self.prompt = read_txt_file(path=self.video_search_keyword_prompt_path)

    def _split_script(self, script: str) -> List[str]:
        """Split script into sentences with total words less than split_length."""
        sentences = re.split(r'(?<=[.!?])\s+', script.strip())
        batches = []
        current_batch = []
        current_word_count = 0

        for sentence in sentences:
            word_count = len(sentence.split())
            if current_word_count + word_count > self.split_length and current_batch:
                batches.append(' '.join(current_batch))
                current_batch = [sentence]
                current_word_count = word_count
            else:
                current_batch.append(sentence)
                current_word_count += word_count

        if current_batch:
            batches.append(' '.join(current_batch))
        
        return batches

    def generate_raw_text_keywords(self, script: str) -> str:
        """Call OpenAI API to generate keyword segments for the script."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                presence_penalty=self.presence_penalty,
                messages=[
                    {"role": "system", "content": self.prompt},
                    {"role": "user", "content": script}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise ValueError(f"OpenAI API call failed: {str(e)}")

    def format_llm_result(self, openai_text: str) -> List[Tuple[str, List[str]]]:
        """Format LLM text result into a structured list of (text, [keywords]) tuples."""
        blocks = re.split(r'---+', openai_text)
        results = []

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            text_match = re.search(r'text:\s*(.*?)(?=\s*keywords:)', block, re.DOTALL)
            keywords_match = re.search(r'keywords:\s*(.*)', block, re.DOTALL)
            if not text_match or not keywords_match:
                continue

            sentence_text = text_match.group(1).strip()
            keywords = [kw.strip() for kw in keywords_match.group(1).split(',')]
            results.append((sentence_text, keywords))

        return results

    def _normalize_text(self, text: str) -> str:
        """Normalize text for better matching by removing extra spaces, punctuation, and special characters."""
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'[\n\r\t]+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip().lower()

    def time_mapping(self, formatted: List[Tuple[str, List[str]]], captions: List[Tuple[Tuple[float, float], str]]) -> List[Tuple[float, float, List[str]]]:
        """Map formatted LLM output to timestamped captions, merging consecutive identical keywords."""
        if not formatted or not captions:
            return []

        sentence_indices = []
        j = 0
        for caption_idx, ((start, end), caption_text) in enumerate(captions):
            caption_text = self._normalize_text(caption_text)
            found = False
            for i in range(j, min(j + 2, len(formatted))):
                sentence = self._normalize_text(formatted[i][0])
                if caption_text in sentence:
                    sentence_indices.append(i + 1)
                    j = i
                    found = True
                    break
            if not found:
                sentence_indices.append("black")
                j = min(j + 1, len(formatted) - 1)

        if not sentence_indices:
            return []

        n = len(formatted)
        processed_indices = sentence_indices.copy()
        processed_indices[0] = 1
        if len(processed_indices) > 1:
            processed_indices[-1] = n

        for i in range(len(processed_indices)):
            if processed_indices[i] == "black":
                last_valid = processed_indices[i-1] if i > 0 else 1
                next_valid = None
                for k in range(i + 1, len(processed_indices)):
                    if processed_indices[k] != "black":
                        next_valid = processed_indices[k]
                        break
                processed_indices[i] = last_valid if next_valid is None else last_valid

        result = []
        current_start = captions[0][0][0]
        current_keywords = formatted[processed_indices[0]-1][1] if processed_indices[0] != "black" else []
        current_idx = processed_indices[0]

        for i in range(1, len(captions)):
            next_idx = processed_indices[i]
            next_keywords = formatted[next_idx-1][1] if next_idx != "black" and 1 <= next_idx <= n else []
            if next_idx != current_idx or i == len(captions)-1:
                end_time = captions[i][0][1] if i == len(captions)-1 else captions[i-1][0][1]
                if current_idx != "black" and 1 <= current_idx <= n:
                    result.append((current_start, end_time, current_keywords, formatted[current_idx-1][0]))
                current_start = captions[i][0][0]
                current_keywords = next_keywords
                current_idx = next_idx

        if len(captions) == 1 and current_idx != "black" and 1 <= current_idx <= n:
            result.append((current_start, captions[0][0][1], current_keywords, formatted[current_idx-1][0]))

        return result

    def get_video_search_queries(
        self,
        script: str,
        captions: List[Tuple[Tuple[float, float], str]]
    ) -> List[Tuple[float, float, List[str]]]:
        """Generate timestamped keyword captions for the script with up to 3 retries."""
        if not script or not isinstance(script, str):
            raise ValueError("Script must be a non-empty string")

        # Split script into batches
        script_batches = self._split_script(script)
        formatted_results = []

        # Process batches concurrently while preserving order
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.generate_raw_text_keywords, batch) for batch in script_batches]
            for future in futures:
                try:
                    raw_output = future.result()
                    formatted = self.format_llm_result(raw_output)
                    formatted_results.append(formatted)
                except Exception as e:
                    raise ValueError(f"Processing batch failed: {str(e)}")

        # Concatenate formatted results
        concatenated_formatted = []
        for formatted in formatted_results:
            concatenated_formatted.extend(formatted)

        # Map to timestamps
        mapping = self.time_mapping(concatenated_formatted, captions)
        return mapping
    
def create_example_captions():
    from src.captions.timed_captions_generator import CaptionGenerator
    from src.Utils.utils import read_txt_file, read_config
    import os

    if os.path.exists('output/audio_tts.wav') and os.path.exists('output/script.txt'):
        test_audio_path = 'output/audio_tts.wav'
        test_script = read_txt_file(path="output/script.txt")
    else:
        test_config = read_config(path='config/test_config.yaml')
        test_audio_path = test_config['test_audio_path']
        test_script = test_config['test_script']

    config = read_config(path='config/config.yaml')
    generator = CaptionGenerator(config)
    corrected_captions = generator.generate_timed_captions(audio_filename=test_audio_path,
                                                          script_text=test_script)

    return corrected_captions, test_script

if __name__ == "__main__":
    config = read_config(path='config/config.yaml')
    generator = VideoKeywordGenerator(config)
    
    captions, test_script = create_example_captions()
    print(f"captions '{captions}':")
    for (start, end), caption in captions:
        print(f"[{start:.2f}s - {end:.2f}s]: {caption}")

    results = generator.get_video_search_queries(test_script, captions)
    
    print("Generated timestamped keywords:")
    for start, end, keywords, sentence in results:
        print(f"[{start:.2f}s - {end:.2f}s]: {keywords} - Sentence: {sentence}")
            
