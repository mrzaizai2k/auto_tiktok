import sys
sys.path.append("")

from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip

# Load file example.mp4 and keep only the subclip from 00:00:10 to 00:00:20
# Reduce the audio volume to 80% of its original volume

clip = (
    VideoFileClip("output/rendered_video.mp4")
    .subclip(0, 2)
)

# Generate a text clip. You can customize the font, color, etc.
txt_clip = TextClip(
    font="font/Neue-Einstellung-Bold.ttf",
    txt="Đắc nhân tâm",
    fontsize=120, color="white",
    stroke_width=4, stroke_color="black", method="label"
    ).set_duration(2)

txt_clip = txt_clip.set_position(["center", 1200])

# Overlay the text clip on the first video clip
final_video = CompositeVideoClip([clip, txt_clip])
final_video.write_videofile("result.mp4")