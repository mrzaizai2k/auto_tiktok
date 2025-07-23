# Text-to-Video AI 🚀

Convert text to video using AI - Mrzaizai2k's way

Inspired by [SamurAIGPT/Text-To-Video-AI](https://github.com/SamurAIGPT/Text-To-Video-AI).

## Setup Instructions

1. Create a `.env` file:
```
OPENAI_KEY=your_openai_key
PEXELS_KEY=your_pexels_key
```

2. Set up environment:
```
conda create -n text2vid python=3.10 -y
conda activate text2vid
pip install -r requirements.txt
```

3. Run the app:
```
python src/app.py "cat brain"
```

Output will be saved to `output/rendered_video.mp4` based on `config/config.yaml`.

## Troubleshooting

**Error**:
```
OSError: MoviePy Error: creation of None failed because of the following error:
convert-im6.q16: attempt to perform an operation not allowed by the security policy...
```

**Solution**:
ImageMagick's security policy is blocking the operation. Modify the policy file:

1. Open `/etc/ImageMagick-6/policy.xml`:
```
sudo nano /etc/ImageMagick-6/policy.xml
```
2. Find and comment out:
```
<!-- <policy domain="path" rights="none" pattern="@*"/> -->
```
3. Save, exit, and restart the app.

## Related Video AI Projects

- [AI Influencer Generator](https://github.com/SamurAIGPT/AI-Influencer-Generator)
- [AI YouTube Shorts Generator](https://github.com/SamurAIGPT/AI-Youtube-Shorts-Generator)
- [Faceless Video Generator](https://github.com/SamurAIGPT/Faceless-Video-Generator)
- [AI B-roll Generator](https://github.com/Anil-matcha/AI-B-roll)
- [Vadoo AI Video Generator](https://www.vadoo.tv/ai-video-generator)
- [Vadoo Text-to-Video AI](https://www.vadoo.tv/text-to-video-ai)
- [Autoshorts AI](https://www.vadoo.tv/autoshorts-ai)
- [Pixverse AI Alternative](https://www.vadoo.tv/pixverse-ai)
- [Hailuo AI Alternative](https://www.vadoo.tv/hailuo-ai)
- [Minimax AI Alternative](https://www.vadoo.tv/minimax-ai)