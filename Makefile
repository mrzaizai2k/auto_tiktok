install:
# 	conda create -n text2vid python=3.10 -y
# 	conda activate text2vid
	pip install -r requirements.txt

run:
	python src/app.py

unit_test:

	python src/script/script_generator.py
	python src/script/description_generator.py
	python src/audio/audio_generator.py
	python src/captions/timed_captions_generator.py
	python src/video/video_search_query_generator.py
	python src/video/background_video_generator.py
	python src/render/render_engine.py
	python src/script/crawl_data.py
	python src/video/image_text_matching.py