import sys
sys.path.append("")

import asyncio
from src.script.script_generator import generate_script
from src.audio.audio_generator import generate_audio
from src.captions.timed_captions_generator import generate_timed_captions
from src.video.background_video_generator import generate_video_url
from src.render.render_engine import get_output_media
from src.video.video_search_query_generator import getVideoSearchQueriesTimed, merge_empty_intervals
import argparse
from src.Utils.utils import read_config, check_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a topic.")
    parser.add_argument("topic", type=str, help="The topic for the video")

    args = parser.parse_args()
    topic = args.topic

    config = read_config(path='config/config.yaml')
    Audio_file_name = config['Audio_file_name']
    output_video_file_name = config['output_video_file_name']
    video_server = config['video_server']

    check_path(Audio_file_name)
    check_path(output_video_file_name)

    script = generate_script(topic)
    print("script: {}".format(script))

    asyncio.run(generate_audio(script, Audio_file_name))

    timed_captions = generate_timed_captions(Audio_file_name)
    print("timed_captions:\n",timed_captions)

    search_terms = getVideoSearchQueriesTimed(script, timed_captions)
    print("search_terms:\n",search_terms)

    background_video_urls = None
    if search_terms is not None:
        background_video_urls = generate_video_url(search_terms, video_server)
        print("background_video_urls\n",background_video_urls)
    else:
        print("No background video")

    background_video_urls = merge_empty_intervals(background_video_urls)
    print("Second background_video_urls\n",background_video_urls)


    if background_video_urls is not None:
        video = get_output_media(Audio_file_name, timed_captions, background_video_urls, 
                                 output_video_file_name=output_video_file_name)
        print(video)
    else:
        print("No video")
