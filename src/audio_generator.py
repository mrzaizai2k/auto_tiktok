import sys
sys.path.append("")

import edge_tts
import os
from typing import Tuple
from cached_path import cached_path
from vinorm import TTSnorm
import soundfile as sf
import numpy as np

from f5_tts.model import DiT

from f5_tts.infer.utils_infer import (
    preprocess_ref_audio_text,
    load_vocoder,
    load_model,
    infer_process,
)

from src.Utils.utils import read_config, timeit



class AudioGenerator:
    """A class to generate audio from text using a Vietnamese TTS model."""

    def __init__(self, config: dict):
        """Initialize the AudioGenerator with model configuration.

        Args:
            config (dict): Configuration dictionary containing model parameters.
        """
        self.config = config["audio_generator"]
        self.vocoder = load_vocoder()
        self.model = load_model(
            DiT,
            self.config["model_params"],
            ckpt_path=str(cached_path(self.config["checkpoint_path"])),
            vocab_file=str(cached_path(self.config["vocab_file"])),
        )

    def post_process(self, text: str) -> str:
        """Clean and normalize input text.

        Args:
            text (str): Input text to process.

        Returns:
            str: Processed text.
        """
        text = f" {text} "
        text = text.replace(" .. ", " . ")
        text = text.replace(" , , ", " , ")
        text = text.replace(" ,, ", " , ")
        text = text.replace('"', "")
        text = text.replace(" . . ", " . ")
        return " ".join(text.split())
    
    def pre_process(self, text: str) -> str:
        """Clean and preprocess input text.

        Args:
            text (str): Input text to process.

        Returns:
            str: Processed text.
        """
        text = f" {text} "
        text = text.replace('...', ".")
        text = text.replace('..', ".")
        return  text.strip()


    def export_wav(self, wav: np.ndarray, file_wave: str) -> None:
        """Save audio waveform to a WAV file.

        Args:
            wav (np.ndarray): Audio waveform.
            file_wave (str): Path to save the WAV file.
        """
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(file_wave), exist_ok=True)
        # Correct parameter order: (filename, data, samplerate)
        sf.write(file_wave, wav, self.config.get("target_sample_rate", 24000))

    @timeit
    def infer_tts(self, text: str) -> Tuple[Tuple[int, np.ndarray], str]:
        """Generate audio from reference audio and text, saving the audio file.

        Args:
            ref_audio_path (str): Path to reference audio file.
            gen_text (str): Text to convert to speech.
            speed (float): Speech speed multiplier. Defaults to 1.0.

        Returns:
            Tuple[Tuple[int, np.ndarray], str]: Generated audio (sample rate, waveform) and audio file path.

        Raises:
            ValueError: If input validation fails or inference errors occur.
        """
        ref_audio_path = self.config.get("ref_audio_path")
        if not ref_audio_path:
            raise ValueError("Reference audio file path is required.")
        if not text.strip():
            raise ValueError("Text content to generate voice is required.")
        if len(text.split()) > 1000:
            raise ValueError("Text content must be less than 1000 words.")

        try:
            preprocessed_text =  self.post_process(TTSnorm(self.pre_process(text))).lower()
            ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_path, "")
            final_wave, final_sample_rate, spectrogram = infer_process(
                ref_audio, ref_text.lower(), preprocessed_text, 
                self.model, self.vocoder, speed=self.config.get("speed", 1.0),
            )
            # Ensure final_wave is a 1D array for mono audio
            if final_wave.ndim > 1:
                final_wave = final_wave.squeeze()

            # Use the configured output audio path instead of temporary file
            audio_path = self.config.get("output_audio_path", "output.wav")
            self.export_wav(final_wave, audio_path)

            return final_sample_rate, final_wave, audio_path
        except Exception as e:
            raise ValueError(f"Error generating voice: {e}")

        
async def generate_audio(config, text:str= "Hello"):
    """Generate audio from text using Edge TTS."""
    audio_config = config['audio_generator']
    output_audio_path = audio_config['output_audio_path']  # Replace with actual audio file path
    voice = audio_config['voice']  # Replace with actual voice name, e.g., "vi-VN-NamMinhNeural"
    communicate = edge_tts.Communicate(text,voice=voice) 

    await communicate.save(output_audio_path)


if __name__ == "__main__":
    import asyncio
    config = read_config(path='config/test_config.yaml')
    test_audio_path = config['test_audio_path']  # Replace with actual Vietnamese audio file path
    test_script = config['test_script']  

    # asyncio.run(generate_audio(text = test_script, config=config))
    # print(f"Audio saved to {test_audio_path}")
    from src.Utils.utils import read_txt_file
    test_script = read_txt_file(path = "output/script.txt")[:600]

    config = read_config(path="config/config.yaml")
    generator = AudioGenerator(config)
    
    audio, _, audio_path = generator.infer_tts(text = test_script)
    print(f"Generated audio saved at {audio_path}")




