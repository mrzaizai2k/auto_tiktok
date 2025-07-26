import sys
sys.path.append("")

import edge_tts
from src.Utils.utils import read_config

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

    asyncio.run(generate_audio(text = test_script, config=config))
    print(f"Audio saved to {test_audio_path}")




