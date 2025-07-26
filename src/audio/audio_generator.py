import sys
sys.path.append("")

from src.Utils.utils import read_config
import edge_tts

async def generate_audio(text,outputFilename):
    communicate = edge_tts.Communicate(text,"vi-VN-NamMinhNeural") #vi-VN-HoaiMyNeural #vi-VN-NamMinhNeural
    await communicate.save(outputFilename)


if __name__ == "__main__":
    import asyncio
    config = read_config(path='config/test_config.yaml')
    test_audio_path = config['test_audio_path']  # Replace with actual Vietnamese audio file path
    test_script = config['test_script']  

    asyncio.run(generate_audio(test_script, test_audio_path))
    print(f"Audio saved to {test_audio_path}")




