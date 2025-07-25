import sys
sys.path.append("")

from typing import List, Dict, Any, Tuple
import os
import re
from openai import OpenAI
from dotenv import load_dotenv
from math import inf
from src.Utils.utils import read_config

load_dotenv()



def _normalize(text: str) -> str:
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', ' ', text.lower())).strip()

def _words(text: str) -> List[str]:
    return [w for w in _normalize(text).split() if w]

def _levenshtein(a: str, b: str) -> int:
    # classic DP
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            cur = prev if ca == cb else prev + 1
            cur = min(cur, dp[j] + 1, dp[j - 1] + 1)
            prev, dp[j] = dp[j], cur
    return dp[-1]

def _similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    dist = _levenshtein(a, b)
    return 1.0 - dist / max(len(a), len(b))

def _best_end_caption_index_by_tail(
    last_tail: str,
    captions: List[Tuple[Tuple[float, float], str]],
    start_idx: int,
    tail_size_each_caption: int = 3,
    min_similarity: float = 0.55,
    max_lookahead: int = 200
) -> int:
    """
    Find the caption index whose *tail* best matches the last 3 words (or fewer) of the sentence.
    """
    best_idx = -1
    best_sim = -inf
    last_tail_norm = _normalize(last_tail)

    end_idx = min(len(captions), start_idx + max_lookahead)
    for i in range(start_idx, end_idx):
        cap_text = captions[i][1]
        cap_tail_words = _words(cap_text)[-tail_size_each_caption:]  # last N words of this caption
        cap_tail = " ".join(cap_tail_words)
        sim = _similarity(last_tail_norm, _normalize(cap_tail))
        if sim > best_sim:
            best_sim = sim
            best_idx = i

    if best_sim >= min_similarity:
        return best_idx
    # fallback: still return the best we can find (avoid None); caller can decide to reject & retry
    return best_idx


class VideoKeywordGenerator:
    """Generates visually concrete keywords for video segments based on a plain script (no timestamps)."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config['keyword_enerator']
        self.model_name = config.get('model_name', 'gpt-4o-mini')
        self.temperature = config.get('temperature', 1.0)
        self.video_search_keyword_prompt_path = config.get(
            'video_search_keyword_prompt_path',
            'config/video_search_keyword_prompt.txt'
        )
        
        self.api_key = os.getenv('OPENAI_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_KEY environment variable not set")
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
        last_n_tail_words: int = 3,
        min_similarity: float = 0.7,
        max_lookahead: int = 200
    ) -> List[Tuple[float, float, List[str]]]:
        """
        Map each (sentence, keywords) to timestamps using
        the last N words + Levenshtein similarity to decide the end time.
        """
        results: List[Tuple[float, float, List[str]]] = []
        caption_idx = 0  # pointer to where we are in captions

        for sentence, keywords in formatted_result:
            sent_tokens = _words(sentence)
            if not sent_tokens:
                continue

            # START time: the next available caption's start
            if caption_idx >= len(captions):
                # nothing left to map => break
                break
            start_time = captions[caption_idx][0][0]

            # END time: find by last_n_tail_words matching
            tail = " ".join(sent_tokens[-last_n_tail_words:])
            end_idx = _best_end_caption_index_by_tail(
                last_tail=tail,
                captions=captions,
                start_idx=caption_idx,
                tail_size_each_caption=min(last_n_tail_words, 3),
                min_similarity=min_similarity,
                max_lookahead=max_lookahead
            )

            if end_idx == -1:
                # couldn't map; let validator force a retry
                end_time = captions[-1][0][1]
            else:
                end_time = captions[end_idx][0][1]
                caption_idx = end_idx + 1  # advance pointer

            results.append((round(start_time, 2), round(end_time, 2), keywords))

        return results

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
    config = read_config(path='config/config.yaml')
    from src.captions.timed_captions_generator import CaptionGenerator, correct_timed_captions
    generator = CaptionGenerator(config)
    test_audio = "output/audio_tts_test.wav"  # Replace with actual Vietnamese audio file path

    captions = generator.generate_timed_captions(test_audio)

    test_script = """Bạn Bạn có biết, cuốn sách "Đắc Nhân Tâm" đã giúp triệu hành hàng triệu người thay đổi cuộc đời? Nhưng cuộc hành trình chinh phục lòng người không hề dễ dàng…  
    Chúng ta cùng theo dõi câu chuyện của một người làm nghề bán hàng, Minh, luôn tràn đầy hoài bão. Anh thấy rằng, trong thế giới cạnh tranh, việc kết nối với khách hàng là rất quan trọng. Anh quyết định đọc "Đắc Nhân Tâm" để hiểu rõ hơn về tâm lý con người.  
    Mục tiêu của Minh rất rõ ràng: muốn trở thành nhân viên xuất sắc, tăng doanh số và ghi dấu ấn trong lòng mỗi khách hàng. Nhưng thực tế thì không như mơ... Anh đối mặt với sự lạnh lùng của những khách hàng khó tính. Không khí im lặng, ánh mắt xa lạ khiến Minh bắt đầu chùn bước.  
    Giữa lúc tuyệt vọng, Minh nhớ lại một trong những bài học quý giá: “Hãy quan tâm đến người khác như bạn muốn được quan tâm.” Anh quyết định áp dụng điều đó.  
    Ngày hôm sau, Minh cố gắng lắng nghe và thấu hiểu tâm sự của từng khách hàng. Thế rồi, bất ngờ xảy ra! Khách hàng bắt đầu cười, chia sẻ và không chỉ mua sản phẩm, họ còn muốn nghe thêm những câu chuyện của Minh.  
    Chỉ sau một thời gian ngắn, doanh số của Minh đã tăng vọt!  
    Bài học rút ra từ câu chuyện này thật đơn giản, nhưng sâu sắc: Kết nối với con người là chìa khóa mở mọi cánh cửa thành công.  
    Vậy bạn có bao giờ thử thấu hiểu tâm tư của người khác chưa"""

    # Correct captions
    corrected_captions = correct_timed_captions(test_script, captions)
    
    return corrected_captions, test_script

if __name__ == "__main__":
    try:
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
            
    except Exception as e:
        print(f"Error: {str(e)}")