# Text To Video AI 🔥

Generate video from text using AI

If you wish to add Text to Video into your application, here is an api to create video from text :- https://docs.vadoo.tv/docs/guide/create-an-ai-video

### Youtube Tutorial -> https://www.youtube.com/watch?v=AXo6VfRUgic

### Medium tutorial -> https://medium.com/@anilmatcha/text-to-video-ai-how-to-create-videos-for-free-a-complete-guide-a25c91de50b8

### Demo Video

https://github.com/user-attachments/assets/1e440ace-8560-4e12-850e-c532740711e7

### 🌟 Show Support

If you enjoy using Text to Video AI, we'd appreciate your support with a star ⭐ on our repository. Your encouragement is invaluable and inspires us to continually improve and expand Text to Video AI. Thank you, and happy content creation! 🎉

[![GitHub star chart](https://img.shields.io/github/stars/SamurAIGPT/Text-To-Video-AI?style=social)](https://github.com/SamurAIGPT/Text-To-Video-AI/stargazers)

### Steps to run

Run the following steps

```
export OPENAI_KEY="api-key"
export PEXELS_KEY="pexels-key"

pip install -r requirements.text

python app.py "Topic name"
```

Output will be generated in rendered_video.mp4

### Quick Start

Without going through the installation hastle here is a simple way to generate videos from text

For a simple way to run the code, checkout the [colab link](/Text_to_Video_example.ipynb)

To generate a video, just click on all the cells one by one. Setup your api keys for openai and pexels

## 💁 Contribution

As an open-source project we are extremely open to contributions. To get started raise an issue in Github or create a pull request

### Other useful Video AI Projects

[AI Influencer generator](https://github.com/SamurAIGPT/AI-Influencer-Generator)

[AI Youtube Shorts generator](https://github.com/SamurAIGPT/AI-Youtube-Shorts-Generator/)

[Faceless Video Generator](https://github.com/SamurAIGPT/Faceless-Video-Generator)

[AI B-roll generator](https://github.com/Anil-matcha/AI-B-roll)

[AI video generator](https://www.vadoo.tv/ai-video-generator)

[Text to Video AI](https://www.vadoo.tv/text-to-video-ai)

[Autoshorts AI](https://www.vadoo.tv/autoshorts-ai)

[Pixverse alternative](https://www.vadoo.tv/pixverse-ai)

[Hailuo AI alternative](https://www.vadoo.tv/hailuo-ai)

[Minimax AI alternative](https://www.vadoo.tv/minimax-ai)

FAQ
error
```
Traceback (most recent call last):
  File "/home/mrzaizai2k/code_Bao/Text-To-Video-AI/app.py", line 47, in <module>
    video = get_output_media(SAMPLE_FILE_NAME, timed_captions, background_video_urls, VIDEO_SERVER)
  File "/home/mrzaizai2k/code_Bao/Text-To-Video-AI/utility/render/render_engine.py", line 67, in get_output_media
    text_clip = TextClip(txt=text, fontsize=100, color="white", stroke_width=3, stroke_color="black", method="label")
  File "/home/mrzaizai2k/anaconda3/envs/text2vid/lib/python3.10/site-packages/moviepy/video/VideoClip.py", line 1146, in __init__
    raise IOError(error)
OSError: MoviePy Error: creation of None failed because of the following error:

convert-im6.q16: attempt to perform an operation not allowed by the security policy `@/tmp/tmpk5cbw4_0.txt' @ error/property.c/InterpretImageProperties/3706.
convert-im6.q16: no images defined `PNG32:/tmp/tmprushj0hj.png' @ error/convert.c/ConvertImageCommand/3229.
.
```
solution
The error occurs because ImageMagick's security policy blocks the operation. Update ImageMagick's policy file (/etc/ImageMagick-6/policy.xml or similar) to allow the operation:

Open the policy file: sudo nano /etc/ImageMagick-6/policy.xml
Find the line: <policy domain="path" rights="none" pattern="@*"/>
Comment it out: <!-- <policy domain="path" rights="none" pattern="@*"/> -->
Save and exit.
Restart your application.

