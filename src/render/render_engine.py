import sys
sys.path.append("")

import os
import tempfile
import platform
import subprocess
from typing import List, Tuple, Optional
from moviepy.editor import (AudioFileClip, CompositeVideoClip, CompositeAudioClip,
                            TextClip, VideoFileClip)
# from moviepy.audio.fx.audio_loop import audio_loop
# from moviepy.audio.fx.audio_normalize import audio_normalize
import requests



def download_file(url: str, filename: str) -> bool:
    """
    Downloads a file from the given URL and saves it to the specified filename.
    
    Args:
        url (str): The URL of the file to download
        filename (str): The local path to save the downloaded file
        
    Returns:
        bool: True if download successful, False otherwise
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:91.0) Gecko/20100101 Firefox/91.0"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {url} to {filename}, size: {os.path.getsize(filename)} bytes")
        if os.path.exists(filename):
            print(f"File {filename} exists")
        return True
    except requests.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return False
    except OSError as e:
        print(f"Error writing file {filename}: {e}")
        return False

def search_program(program_name: str) -> Optional[str]:
    """
    Searches for the specified program in the system PATH.
    
    Args:
        program_name (str): Name of the program to search for
        
    Returns:
        Optional[str]: Path to the program if found, None otherwise
    """
    try:
        search_cmd = "where" if platform.system() == "Windows" else "which"
        return subprocess.check_output([search_cmd, program_name]).decode().strip()
    except subprocess.CalledProcessError as e:
        print(f"Error searching for program {program_name}: {e}")
        return None
    except UnicodeDecodeError as e:
        print(f"Error decoding program path for {program_name}: {e}")
        return None

def get_program_path(program_name: str) -> Optional[str]:
    """
    Gets the path to the specified program.
    
    Args:
        program_name (str): Name of the program to find
        
    Returns:
        Optional[str]: Path to the program if found, None otherwise
    """
    try:
        program_path = search_program(program_name)
        return program_path
    except Exception as e:
        print(f"Error getting program path for {program_name}: {e}")
        return None

def get_output_media(
    audio_file_path: str,
    timed_captions: List[Tuple[Tuple[float, float], str]],
    background_video_data: List[Tuple[Tuple[float, float], str]],
    output_video_file_name: str = "rendered_video.mp4"
) -> Optional[str]:
    """
    Creates a composite video with audio, background videos, and captions.
    
    Args:
        audio_file_path (str): Path to the audio file
        timed_captions (List[Tuple[Tuple[float, float], str]]): List of captions with start/end times
        background_video_data (List[Tuple[Tuple[float, float], str]]): List of background videos with start/end times and URLs
        output_video_file_name (str): Name of the output video file
        
    Returns:
        Optional[str]: Path to the output video file if successful, None otherwise
    """
    try:
        magick_path = get_program_path("magick")
        print("magick_path", magick_path)
        if magick_path:
            os.environ['IMAGEMAGICK_BINARY'] = magick_path
        else:
            os.environ['IMAGEMAGICK_BINARY'] = '/usr/bin/convert'
        
        visual_clips = []
        for (t1, t2), video_url in background_video_data:
            try:
                video_filename = tempfile.NamedTemporaryFile(delete=False).name
                if not download_file(video_url, video_filename):
                    continue
                
                video_clip = VideoFileClip(video_filename)
                video_clip = video_clip.set_start(t1)
                video_clip = video_clip.set_end(t2)
                visual_clips.append(video_clip)
            except Exception as e:
                print(f"Error processing video {video_url}: {e}")
                continue
        
        audio_clips = []
        try:
            audio_file_clip = AudioFileClip(audio_file_path)
            audio_clips.append(audio_file_clip)
        except Exception as e:
            print(f"Error loading audio file {audio_file_path}: {e}")
            return None

        for (t1, t2), text in timed_captions:
            try:
                text_clip = TextClip(txt=text, fontsize=70, 
                                     color="white", 
                                     font="font/Neue-Einstellung-Bold.ttf",
                                  stroke_width=2, stroke_color="black", method="label")
                text_clip = text_clip.set_start(t1)
                text_clip = text_clip.set_end(t2)
                text_clip = text_clip.set_position(["center", 1800])
                visual_clips.append(text_clip)
            except Exception as e:
                print(f"Error creating text clip for '{text}': {e}")
                continue

        try:
            video = CompositeVideoClip(visual_clips)
        except Exception as e:
            print(f"Error creating composite video: {e}")
            return None
        
        if audio_clips:
            try:
                audio = CompositeAudioClip(audio_clips)
                video.duration = audio.duration
                video.audio = audio
            except Exception as e:
                print(f"Error processing audio: {e}")
                return None

        try:
            video.write_videofile(output_video_file_name, codec='libx264', 
                                audio_codec='aac', fps=25, preset='veryfast')
        except Exception as e:
            print(f"Error writing video file {output_video_file_name}: {e}")
            return None
        
        for (t1, t2), video_url in background_video_data:
            try:
                video_filename = tempfile.NamedTemporaryFile(delete=False).name
                if os.path.exists(video_filename):
                    os.remove(video_filename)
            except OSError as e:
                print(f"Error cleaning up file {video_filename}: {e}")
                continue

        return output_video_file_name
    except Exception as e:
        print(f"Unexpected error in get_output_media: {e}")
        return None