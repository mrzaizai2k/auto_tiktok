install:
	conda create -n text2vid python=3.10 -y
	conda activate text2vid
	pip install -r requirements.txt

test:
	python src/app.py "Đắc Nhân Tâm"
