import sys
sys.path.append("")
import os
import tempfile
from typing import List, Tuple, Optional, Dict, Any
from moviepy.editor import AudioFileClip, CompositeVideoClip, CompositeAudioClip, TextClip, VideoFileClip
from src.Utils.utils import download_file, read_config

class VideoComposer:
    """Class to create composite videos with optional audio, background videos, and captions."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize VideoComposer with configuration.
        
        Args:
            config: Configuration dictionary with video parameters.
        """
        self.config = config['video_composer']
        self.output_video_path = config.get('output_video_path', 'rendered_video.mp4')
        self.font_path = config.get('font_path', 'font/Neue-Einstellung-Bold.ttf')
        self.font_size = config.get('font_size', 70)
        self.text_color = config.get('text_color', 'white')
        self.stroke_width = config.get('stroke_width', 2)
        self.stroke_color = config.get('stroke_color', 'black')
        self.text_position = config.get('text_position', ['center', 1800])
        self.video_codec = config.get('video_codec', 'libx264')
        self.audio_codec = config.get('audio_codec', 'aac')
        self.fps = config.get('fps', 25)
        self.preset = config.get('preset', 'veryfast')

    def _create_video_clips(self, background_video_data: List[Tuple[Tuple[float, float], str]]) -> List[VideoFileClip]:
        """Create video clips from background video data.
        
        Args:
            background_video_data: List of tuples with time ranges and video URLs.
            
        Returns:
            List of video clips.
        """
        visual_clips = []
        for (t1, t2), video_url in background_video_data:
            try:
                video_filename = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                if not download_file(video_url, video_filename):
                    continue
                video_clip = VideoFileClip(video_filename).set_start(t1).set_end(t2)
                visual_clips.append(video_clip)
            except Exception as e:
                print(f"Error processing video {video_url}: {e}")
        return visual_clips

    def _create_audio_clip(self, audio_file_path: Optional[str]) -> Optional[List[AudioFileClip]]:
        """Create audio clip if audio file path is provided.
        
        Args:
            audio_file_path: Path to audio file or None.
            
        Returns:
            List containing audio clip or None.
        """
        if not audio_file_path:
            return None
        try:
            return [AudioFileClip(audio_file_path)]
        except Exception as e:
            print(f"Error loading audio file {audio_file_path}: {e}")
            return None

    def _create_text_clips(self, timed_captions: Optional[List[Tuple[Tuple[float, float], str]]]) -> List[TextClip]:
        """Create text clips for captions if provided.
        
        Args:
            timed_captions: List of tuples with time ranges and caption text or None.
            
        Returns:
            List of text clips.
        """
        if not timed_captions:
            return []
        text_clips = []
        for (t1, t2), text in timed_captions:
            try:
                text_clip = TextClip(
                    txt=text,
                    fontsize=self.font_size,
                    color=self.text_color,
                    font=self.font_path,
                    stroke_width=self.stroke_width,
                    stroke_color=self.stroke_color,
                    method="label"
                ).set_start(t1).set_end(t2).set_position(self.text_position)
                text_clips.append(text_clip)
            except Exception as e:
                print(f"Error creating text clip for '{text}': {e}")
        return text_clips

    def _cleanup_temp_files(self, background_video_data: List[Tuple[Tuple[float, float], str]]) -> None:
        """Clean up temporary video files.
        
        Args:
            background_video_data: List of tuples with time ranges and video URLs.
        """
        for _ in background_video_data:
            try:
                video_filename = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                if os.path.exists(video_filename):
                    os.remove(video_filename)
            except OSError as e:
                print(f"Error cleaning up file {video_filename}: {e}")

    def compose_video(
        self,
        background_video_urls: List[Tuple[Tuple[float, float], str]],
        audio_file_path: Optional[str] = None,
        timed_captions: Optional[List[Tuple[Tuple[float, float], str]]] = None
    ) -> Optional[str]:
        """Compose video with background videos, optional audio, and captions.
        
        Args:
            background_video_data: List of tuples with time ranges and video URLs.
            audio_file_path: Path to audio file or None.
            timed_captions: List of tuples with time ranges and caption text or None.
            
        Returns:
            Path to output video file or None if failed.
        """
        try:
            visual_clips = self._create_video_clips(background_video_urls)
            if not visual_clips:
                print("No valid video clips created")
                return None

            visual_clips.extend(self._create_text_clips(timed_captions))
            video = CompositeVideoClip(visual_clips)

            audio_clips = self._create_audio_clip(audio_file_path)
            if audio_clips:
                try:
                    audio = CompositeAudioClip(audio_clips)
                    video.duration = audio.duration
                    video.audio = audio
                except Exception as e:
                    print(f"Error processing audio: {e}")
                    return None

            try:
                video.write_videofile(
                    self.output_video_path,
                    codec=self.video_codec,
                    audio_codec=self.audio_codec,
                    fps=self.fps,
                    preset=self.preset
                )
            except Exception as e:
                print(f"Error writing video file {self.output_video_path}: {e}")
                return None

            self._cleanup_temp_files(background_video_urls)
            return self.output_video_path

        except Exception as e:
            print(f"Unexpected error in compose_video: {e}")
            return None

if __name__ == "__main__":
    config = read_config(path='config/config.yaml')
    composer = VideoComposer(config)
    
    # Example test data
    background_videos = [
        ((0, 5), "https://cdn.pixabay.com/video/2024/08/30/228847_medium.mp4"),
        ((5, 10), "https://videos.pexels.com/video-files/7550336/7550336-hd_1080_1920_30fps.mp4")
    ]
    timed_captions = [
        ((0, 5), "Hello, World!"),
        ((5, 10), "Welcome to Video!")
    ]

    test_config = read_config(path='config/test_config.yaml')
    test_audio_path = test_config['test_audio_path']  # Replace with actual Vietnamese audio file path
    
    
    output = composer.compose_video(
        background_video_urls=background_videos,
        audio_file_path=test_audio_path,
        timed_captions=timed_captions
    )
    print(f"Video created: {output}")