import sys
sys.path.append("")

from typing import Dict, Any, List, Tuple
from whisper_timestamped import load_model, transcribe_timestamped
import re
import os
from src.Utils.utils import read_config, levenshtein_distance
import librosa

class CaptionGenerator:
    """Generates timed captions from audio using Whisper model and creates SRT file if configured."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize CaptionGenerator with configuration.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary from config.yaml
            
        Raises:
            ValueError: If required config parameters are missing
            FileNotFoundError: If Whisper model fails to load
        """
        self.config = config['caption_generator']
        self.model_size = self.config.get('whisper_model_size', 'base')
        self.max_caption_size = self.config.get('max_caption_size', 15)
        self.consider_punctuation = self.config.get('consider_punctuation', False)
        self.language = self.config.get('language', 'vi')  # Vietnamese as default
        self.srt_file_path = self.config.get('srt_file_path', None)
        
        try:
            self.whisper_model = load_model(self.model_size)
        except Exception as e:
            raise ValueError(f"Failed to load Whisper model: {str(e)}")

    def get_video_length(self, audio_filename: str) -> float:
        """Get the duration of the audio file in seconds.
        
        Args:
            audio_filename (str): Path to audio file
            
        Returns:
            float: Duration of the audio file in seconds
            
        Raises:
            FileNotFoundError: If audio file is not found
            ValueError: If duration cannot be determined
        """
        try:
            if not os.path.exists(audio_filename):
                raise FileNotFoundError(f"Audio file not found: {audio_filename}")
            return librosa.get_duration(filename=audio_filename)
        except Exception as e:
            raise ValueError(f"Failed to get video length: {str(e)}")

    def split_words_by_size(self, words: List[str], max_caption_size: int) -> List[str]:
        """Split words into captions based on maximum caption size.
        
        Args:
            words (List[str]): List of words to split
            max_caption_size (int): Maximum size of each caption
            
        Returns:
            List[str]: List of captions
        """
        half_caption_size = max_caption_size / 2
        captions = []
        while words:
            caption = words[0]
            words = words[1:]
            while words and len(caption + ' ' + words[0]) <= max_caption_size:
                caption += ' ' + words[0]
                words = words[1:]
                if len(caption) >= half_caption_size and words:
                    break
            captions.append(caption)
        return captions

    def clean_word(self, word: str) -> str:
        """Clean word by removing unwanted characters.
        
        Args:
            word (str): Word to clean
            
        Returns:
            str: Cleaned word
        """
        return re.sub(r'[^\w\s\-_"\'\']', '', word)

    def get_timestamp_mapping(self, whisper_analysis: Dict[str, Any]) -> Dict[Tuple[int, int], float]:
        """Create mapping of word positions to timestamps.
        
        Args:
            whisper_analysis (Dict[str, Any]): Whisper transcription analysis
            
        Returns:
            Dict[Tuple[int, int], float]: Mapping of position ranges to timestamps
        """
        index = 0
        location_to_timestamp = {}
        for segment in whisper_analysis['segments']:
            for word in segment['words']:
                new_index = index + len(word['text']) + 1
                location_to_timestamp[(index, new_index)] = word['end']
                index = new_index
        return location_to_timestamp

    def interpolate_time_from_dict(self, word_position: int, time_dict: Dict[Tuple[int, int], float]) -> float:
        """Interpolate timestamp for a word position.
        
        Args:
            word_position (int): Position of word in text
            time_dict (Dict[Tuple[int, int], float]): Timestamp mapping
            
        Returns:
            float: Interpolated timestamp or None if not found
        """
        for (start, end), value in time_dict.items():
            if start <= word_position <= end:
                return value
        return None

    def generate_timed_captions(self, audio_filename: str, script_text: str = None) -> List[Tuple[Tuple[float, float], str]]:
        """Generate timed captions from audio file and optionally correct with script text.
        
        Args:
            audio_filename (str): Path to audio file
            script_text (str, optional): Script text for correction
            
        Returns:
            List[Tuple[Tuple[float, float], str]]: List of (time range, caption) pairs
            
        Raises:
            FileNotFoundError: If audio file is not found
            ValueError: If transcription or processing fails
        """
        try:
            if not os.path.exists(audio_filename):
                raise FileNotFoundError(f"Audio file not found: {audio_filename}")

            whisper_analysis = transcribe_timestamped(
                self.whisper_model,
                audio_filename,
                verbose=False,
                fp16=False,
                language=self.language
            )
            
            word_location_to_time = self.get_timestamp_mapping(whisper_analysis)
            position = 0
            start_time = 0
            captions = []
            text = whisper_analysis['text']
            
            if self.consider_punctuation:
                sentences = re.split(r'(?<=[.!?]) +', text)
                words = [word for sentence in sentences for word in self.split_words_by_size(sentence.split(), self.max_caption_size)]
            else:
                words = text.split()
                words = [self.clean_word(word) for word in self.split_words_by_size(words, self.max_caption_size)]
            
            video_length = self.get_video_length(audio_filename)
            for i, word in enumerate(words):
                position += len(word) + 1
                end_time = self.interpolate_time_from_dict(position, word_location_to_time)
                if end_time and word:
                    if i == len(words) - 1:  # Last caption
                        captions.append(((start_time, video_length), word))
                    else:
                        captions.append(((start_time, end_time), word))
                        start_time = end_time
                    

            if script_text:
                captions = self.correct_timed_captions(script_text, captions)
            
            if self.srt_file_path:
                self.generate_srt_file(captions, self.srt_file_path)
            
            return captions
            
        except Exception as e:
            raise ValueError(f"Failed to generate captions: {str(e)}")

    def correct_timed_captions(self, script_text: str, captions: List[Tuple[Tuple[float, float], str]]) -> List[Tuple[Tuple[float, float], str]]:
        """Correct timed captions to match the provided script text while preserving original timestamps.
        
        Args:
            script_text (str): The correct script text to align captions with
            captions (List[Tuple[Tuple[float, float], str]]): List of timed captions
            
        Returns:
            List[Tuple[Tuple[float, float], str]]: Corrected list of (time range, caption) pairs
            
        Raises:
            ValueError: If inputs are invalid or processing fails
        """
        try:
            if not script_text or not isinstance(script_text, str):
                raise ValueError("Script text must be a non-empty string")
            if not captions or not isinstance(captions, list):
                raise ValueError("Captions must be a non-empty list of timed caption tuples")

            cleaned_script = re.sub(r'[^\w\s]', '', script_text).strip()
            script_words = cleaned_script.split()
            if not script_words:
                raise ValueError("Script text contains no valid words after cleaning")

            corrected_captions = []
            for timestamp, caption in captions:
                caption_words = caption.split()
                num_words = len(caption_words)
                if num_words == 0:
                    corrected_captions.append((timestamp, caption))
                    continue

                best_match = ""
                best_score = -1
                best_start_idx = 0

                for i in range(len(script_words) - num_words + 1):
                    candidate_segment = ' '.join(script_words[i:i + num_words])
                    similarity = levenshtein_distance(caption.lower(), candidate_segment.lower())
                    max_length = max(len(caption), len(candidate_segment))
                    similarity_score = 1 - (similarity / max_length) if max_length > 0 else 0
                    
                    if similarity_score > best_score:
                        best_score = similarity_score
                        best_match = candidate_segment
                        best_start_idx = i

                if best_score < 0.4:
                    corrected_captions.append((timestamp, caption))
                else:
                    corrected_captions.append((timestamp, best_match))

            return corrected_captions

        except Exception as e:
            raise ValueError(f"Failed to correct captions: {str(e)}")

    def generate_srt_file(self, captions: List[Tuple[Tuple[float, float], str]], srt_file_path: str) -> None:
        """Generate an SRT file from timed captions.
        
        Args:
            captions (List[Tuple[Tuple[float, float], str]]): List of (time range, caption) pairs
            srt_file_path (str): Path to save the SRT file
            
        Raises:
            ValueError: If SRT file creation fails
        """
        try:
            with open(srt_file_path, 'w', encoding='utf-8') as f:
                for i, ((start, end), caption) in enumerate(captions, 1):
                    start_time = self._format_srt_time(start)
                    end_time = self._format_srt_time(end)
                    f.write(f"{i}\n{start_time} --> {end_time}\n{caption}\n\n")
        except Exception as e:
            raise ValueError(f"Failed to create SRT file: {str(e)}")

    def _format_srt_time(self, seconds: float) -> str:
        """Format time in seconds to SRT time format (HH:MM:SS,mmm).
        
        Args:
            seconds (float): Time in seconds
            
        Returns:
            str: Formatted time string
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

if __name__ == "__main__":

    config = read_config(path='config/config.yaml')
    generator = CaptionGenerator(config)

    test_config = read_config(path='config/test_config.yaml')

    # test_audio_path = test_config['test_audio_path']
    # test_script = test_config['test_script']

    test_audio_path = 'output/audio_tts.wav'
    with open('output/script.txt', 'r', encoding='utf-8') as f:
        test_script = f.read().strip()

    captions = generator.generate_timed_captions(test_audio_path, test_script)
    print(f"Captions:")
    for (start, end), caption in captions:
        print(f"[{start:.2f}s - {end:.2f}s]: {caption}")