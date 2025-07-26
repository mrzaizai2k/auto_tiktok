import sys
sys.path.append("")

import asyncio
from src.script.script_generator import ScriptGenerator
from src.audio.audio_generator import generate_audio
from src.captions.timed_captions_generator import CaptionGenerator, correct_timed_captions
from src.video.background_video_generator import VideoSearch
from src.render.render_engine import VideoComposer
from src.video.video_search_query_generator import VideoKeywordGenerator
import argparse
from src.Utils.utils import read_config, check_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a topic.")
    parser.add_argument("topic", type=str, help="The topic for the video")

    args = parser.parse_args()
    topic = args.topic
    config = read_config(path='config/config.yaml')
    output_audio_path = config['output_audio_path']
    output_video_path = config['output_video_path']
    video_server = config['video_server']

    check_path(output_audio_path)
    check_path(output_video_path)

    generator = ScriptGenerator(config)
    script = generator.generate_script(topic)

    print("script: {}".format(script))

    asyncio.run(generate_audio(text=script, config=config))

    caption_generator = CaptionGenerator(config)
    timed_captions = caption_generator.generate_timed_captions(output_audio_path)
    timed_captions = correct_timed_captions(script, timed_captions)
    print("timed_captions:\n",timed_captions)

    keyword_generator = VideoKeywordGenerator(config)
    search_terms = keyword_generator.get_video_search_queries(script=script, captions=timed_captions)
    print("search_terms:\n",search_terms)

    background_video_urls = None
    if search_terms is not None:
        video_search = VideoSearch(config)
        background_video_urls = video_search.generate_video_urls(search_terms, video_server)
        print("background_video_urls\n",background_video_urls)
    else:
        print("No background video")

    # background_video_urls = merge_empty_intervals(background_video_urls)
    # print("Second background_video_urls\n",background_video_urls)


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
