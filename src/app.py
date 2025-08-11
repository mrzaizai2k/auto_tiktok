import sys
sys.path.append("")

import asyncio
from src.script.description_generator import DescriptionGenerator
from src.script.script_generator import ScriptGenerator
from src.audio_generator import generate_audio, AudioGenerator
from src.captions.timed_captions_generator import CaptionGenerator
from src.video.background_video_generator import VideoSearch
from src.render.render_engine import VideoComposer
from src.video.video_search_query_generator import VideoKeywordGenerator
from src.Utils.utils import read_config, check_path

if __name__ == "__main__":

    topic = "sách Đắc Nhân Tâm" 

    config = read_config(path='config/config.yaml')
    output_audio_path = config['output_audio_path']
    output_video_path = config['output_video_path']

    check_path(output_audio_path)
    check_path(output_video_path)

    # script_generator = ScriptGenerator(config)
    # script = script_generator.generate_script(topic)

    with open('output/script.txt', 'r', encoding='utf-8') as file:
        script = file.read().strip()

    description_generator = DescriptionGenerator(config)
    description = description_generator.generate_description(script)
    

    print("script: {}".format(script))

    audio_generator = AudioGenerator(config)
    try:
        audio, _, audio_path = audio_generator.infer_tts(text = script)
    except Exception as e:
        asyncio.run(generate_audio(text=script, config=config))

    caption_generator = CaptionGenerator(config)
    timed_captions = caption_generator.generate_timed_captions(audio_filename=output_audio_path,
                                                               script_text=script)
    print("timed_captions:\n",timed_captions)

    keyword_generator = VideoKeywordGenerator(config)
    search_terms = keyword_generator.get_video_search_queries(script=script, captions=timed_captions)
    print("search_terms:\n",search_terms)

    background_video_urls = None
    if search_terms is not None:
        video_search = VideoSearch(config)
        background_video_urls = video_search.generate_video_urls(search_terms)
        print("background_video_urls\n",background_video_urls)
    else:
        print("No background video")

    if background_video_urls is not None:
        composer = VideoComposer(config)
        video = composer.compose_video(
            background_video_urls=background_video_urls,
            audio_file_path=output_audio_path,
            timed_captions=timed_captions
        )

        print(video)
    else:
        print("No video")
