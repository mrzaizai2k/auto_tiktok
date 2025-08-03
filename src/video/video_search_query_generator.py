import sys
sys.path.append("")

from typing import List, Dict, Any, Tuple
import os
import re
from openai import OpenAI
from dotenv import load_dotenv
from math import inf
from difflib import SequenceMatcher
from src.Utils.utils import read_config

load_dotenv()


def _normalize_text(text: str) -> str:
    """Normalize text for better matching by removing extra spaces and punctuation."""
    # Remove punctuation and normalize spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

def _words(text: str) -> List[str]:
    """Extract words from text."""
    normalized = _normalize_text(text)
    return normalized.split() if normalized else []

def _levenshtein_similarity(s1: str, s2: str) -> float:
    """Calculate similarity using SequenceMatcher (similar to Levenshtein but faster)."""
    return SequenceMatcher(None, s1, s2).ratio()

def fill_time_gaps(
    results: List[Tuple[float, float, List[str]]],
    captions: List[Tuple[Tuple[float, float], str]],
    gap_threshold: float = 0.1
) -> List[Tuple[float, float, List[str]]]:
    """
    Fill gaps between time spans by extending previous spans or connecting to next spans.
    
    Args:
        results: List of (start_time, end_time, keywords) tuples
        captions: Original captions for reference
        gap_threshold: Minimum gap size to consider filling (in seconds)
    
    Returns:
        List of (start_time, end_time, keywords) with gaps filled
    """
    if not results:
        return results
    
    filled_results = []
    
    for i in range(len(results)):
        start_time, end_time, keywords = results[i]
        
        # For the first span, ensure it starts from the beginning
        if i == 0 and captions:
            first_caption_start = captions[0][0][0]
            if start_time > first_caption_start + gap_threshold:
                start_time = first_caption_start
                print(f"Extended first span to start from {first_caption_start}")
        
        # Check for gap with next span
        if i < len(results) - 1:
            next_start_time = results[i + 1][0]
            gap_size = next_start_time - end_time
            
            if gap_size > gap_threshold:
                # Extend current span to fill the gap
                end_time = next_start_time
                print(f"Filled gap of {gap_size:.2f}s by extending span {i+1} to {end_time}")
        
        # For the last span, ensure it goes to the very end
        elif captions:  # This is the last span
            last_caption_end = captions[-1][0][1]
            if end_time < last_caption_end - gap_threshold:
                end_time = last_caption_end
                print(f"Extended last span to end at {last_caption_end}")
        
        filled_results.append((round(start_time, 2), round(end_time, 2), keywords))
    
    return filled_results


def _find_sentence_boundaries(
    sentence: str,
    captions: List[Tuple[Tuple[float, float], str]],
    start_search_idx: int = 0,
    window_size: int = 10,
    min_similarity: float = 0.6
) -> Tuple[int, int]:
    """
    Find start and end caption indices for a sentence using sliding window + similarity.
    
    Returns:
        Tuple of (start_idx, end_idx) or (-1, -1) if not found
    """
    sentence_words = _words(sentence)
    if not sentence_words:
        return -1, -1
    
    sentence_text = " ".join(sentence_words)
    best_start = -1
    best_end = -1
    best_score = 0
    
    # Search within a sliding window to avoid matching too far away
    search_end = min(len(captions), start_search_idx + window_size * 2)
    
    for start_idx in range(start_search_idx, search_end):
        # Try different window sizes for matching
        for window in range(1, min(window_size, len(captions) - start_idx + 1)):
            end_idx = start_idx + window - 1
            
            # Combine caption texts in this window
            window_texts = []
            for i in range(start_idx, end_idx + 1):
                window_texts.append(captions[i][1])
            
            combined_caption = " ".join(window_texts)
            combined_caption_normalized = _normalize_text(combined_caption)
            
            # Calculate similarity
            similarity = _levenshtein_similarity(sentence_text, combined_caption_normalized)
            
            if similarity > best_score and similarity >= min_similarity:
                best_score = similarity
                best_start = start_idx
                best_end = end_idx
    
    return best_start, best_end

def _find_best_match_progressive(
    sentence: str,
    captions: List[Tuple[Tuple[float, float], str]],
    start_search_idx: int = 0,
    max_window: int = 15,
    min_similarity: float = 0.5
) -> Tuple[int, int, float]:
    """
    Progressive search: start with high similarity and small window, 
    then gradually relax constraints.
    """
    sentence_words = _words(sentence)
    if not sentence_words:
        return -1, -1, 0.0
    
    sentence_text = " ".join(sentence_words)
    
    # Progressive search with decreasing similarity thresholds
    similarity_thresholds = [0.8, 0.7, 0.6, min_similarity]
    window_sizes = [5, 8, 12, max_window]
    
    for sim_threshold in similarity_thresholds:
        for window_size in window_sizes:
            start_idx, end_idx = _find_sentence_boundaries(
                sentence, captions, start_search_idx, window_size, sim_threshold
            )
            
            if start_idx != -1:
                # Calculate actual similarity for the found match
                window_texts = [captions[i][1] for i in range(start_idx, end_idx + 1)]
                combined_caption = _normalize_text(" ".join(window_texts))
                actual_similarity = _levenshtein_similarity(sentence_text, combined_caption)
                return start_idx, end_idx, actual_similarity
    
    return -1, -1, 0.0


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
        
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = OpenAI(api_key=self.api_key)
        
        try:
            with open(self.video_search_keyword_prompt_path, 'r', encoding='utf-8') as file:
                self.prompt = file.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt file not found at {self.video_search_keyword_prompt_path}")

    # ----------------- OpenAI Call -------------------
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

    # ----------------- Formatting -------------------
    def format_llm_result(self, openai_text: str) -> List[Tuple[str, List[str]]]:
        """
        Format LLM text result into a structured list of (text, [keywords]) tuples.
        
        Example output:
        [
            ("Sentence 1", ["keyword1", "keyword2", "keyword3"]),
            ("Sentence 2", ["keyword4", "keyword5", "keyword6"])
        ]
        """
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

    # ----------------- Time Mapping -------------------
    def time_mapping(
            self,
            formatted_result: List[Tuple[str, List[str]]],
            captions: List[Tuple[Tuple[float, float], str]],
            *,
            min_similarity: float = 0.5,
            max_window: int = 15,
            overlap_tolerance: float = 0.5
        ) -> List[Tuple[float, float, List[str]]]:
        """
        Map each (sentence, keywords) to timestamps using sliding window + Levenshtein similarity.
        Includes gap filling to eliminate empty time spans.
        
        Args:
            formatted_result: List of (sentence, keywords) tuples
            captions: List of ((start_time, end_time), caption_text) tuples
            min_similarity: Minimum similarity threshold for matching
            max_window: Maximum window size for searching
            overlap_tolerance: Tolerance for overlapping matches (in seconds)
        
        Returns:
            List of (start_time, end_time, keywords) tuples with no gaps
        """
        if not formatted_result or not captions:
            return []
        
        results: List[Tuple[float, float, List[str]]] = []
        last_used_idx = 0
        
        for i, (sentence, keywords) in enumerate(formatted_result):
            if not sentence.strip():
                continue
            
            # Check if this is the last sentence
            is_last_sentence = (i == len(formatted_result) - 1)
            
            # Find the best match for this sentence
            start_idx, end_idx, similarity = _find_best_match_progressive(
                sentence=sentence,
                captions=captions,
                start_search_idx=last_used_idx,
                max_window=max_window,
                min_similarity=min_similarity
            )
            
            if start_idx == -1:
                # Fallback: couldn't find good match
                if results:
                    # Use time after the last result
                    last_end_time = results[-1][1]
                    # Try to estimate duration based on sentence length
                    estimated_duration = max(2.0, len(_words(sentence)) * 0.3)
                    start_time = last_end_time
                    
                    # For last sentence, extend to the very end of captions
                    if is_last_sentence:
                        end_time = captions[-1][0][1]
                    else:
                        end_time = start_time + estimated_duration
                else:
                    # First sentence, use beginning of captions
                    start_time = captions[0][0][0]
                    
                    # If it's also the last sentence (only one sentence), use full duration
                    if is_last_sentence:
                        end_time = captions[-1][0][1]
                    else:
                        end_time = captions[min(5, len(captions)-1)][0][1]
                
                print(f"Warning: Could not find good match for sentence {i+1}: '{sentence[:50]}...'")
                print(f"Using fallback timing: {start_time}-{end_time}")
            else:
                start_time = captions[start_idx][0][0]
                
                # For the last sentence, always extend to the end of all captions
                if is_last_sentence:
                    end_time = captions[-1][0][1]
                    print(f"Last sentence - extending to final caption time: {end_time}")
                else:
                    end_time = captions[end_idx][0][1]
                
                # Update the search starting point for next sentence
                # Allow some overlap to handle cases where sentences share some words
                overlap_captions = int(overlap_tolerance / 0.5)  # Rough estimate
                last_used_idx = max(last_used_idx, end_idx - overlap_captions)
                
                # print(f"Matched sentence {i+1} (similarity: {similarity:.2f}): "
                #     f"'{sentence[:30]}...' -> {start_time}-{end_time}s")
            
            results.append((
                round(start_time, 2),
                round(end_time, 2),
                keywords
            ))
        
        # Fill gaps to eliminate empty/black spans
        filled_results = fill_time_gaps(results, captions)
        
        return filled_results
    # ----------------- Validation -------------------
    def validate_mapping(self, mapping: List[Tuple[float, float, List[str]]]) -> bool:
        """
        Validate mapping to ensure:
        - No duplicate time ranges.
        - Non-empty keywords.
        """
        seen_times = set()
        for start, end, keywords in mapping:
            if (start, end) in seen_times:
                return False
            seen_times.add((start, end))
            if not keywords or any(not k.strip() for k in keywords):
                return False
        return True

    def validate_coverage(
        self,
        mapping: List[Tuple[float, float, List[str]]],
        captions: List[Tuple[Tuple[float, float], str]]
    ) -> bool:
        """
        Ensure the mapping covers the entire range of the captions.
        The last mapping end time should match the last caption's end time (±0.3s tolerance).
        """
        if not mapping:
            return False

        last_caption_end = captions[-1][0][1]
        last_mapping_end = mapping[-1][1]

        if abs(last_caption_end - last_mapping_end) > 0.3:
            # Coverage issue detected
            print(f"⚠️ Coverage mismatch: last mapping ends at {last_mapping_end:.2f}s "
                  f"but captions end at {last_caption_end:.2f}s.")
            return False

        return True

    def get_video_search_queries(
        self,
        script: str,
        captions: List[Tuple[Tuple[float, float], str]]
    ) -> List[Tuple[float, float, List[str]]]:
        """Generate timestamped keyword captions for the script with up to 3 retries."""
        if not script or not isinstance(script, str):
            raise ValueError("Script must be a non-empty string")

        for attempt in range(3):
            raw_output = self.generate_raw_text_keywords(script)
            formatted = self.format_llm_result(raw_output)
            mapping = self.time_mapping(formatted, captions)

            if self.validate_mapping(mapping) and self.validate_coverage(mapping, captions):
                return mapping
            else:
                print(f"⚠️ Validation or coverage failed (attempt {attempt+1}). Retrying with new LLM call...")

        raise ValueError("Failed to generate valid video search queries after 3 attempts.")


def create_example_captions():
    test_config = read_config(path='config/test_config.yaml')
    from src.captions.timed_captions_generator import CaptionGenerator

    # test_audio_path = test_config['test_audio_path']  # Replace with actual Vietnamese audio file path
    # test_script = test_config['test_script']  

    test_audio_path = 'output/audio_tts.wav'  # Example path
    with open("output/script.txt", 'r', encoding='utf-8') as f:
        test_script = f.read().strip()

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
    for start, end, keywords in results:
        print(f"[{start:.2f}s - {end:.2f}s]: {keywords}")
            
