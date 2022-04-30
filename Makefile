
PYTHON ?= python3.8

venv/bin/activate: requirements.txt
	{PYTHON} -m venv venv
	./venv/bin/python -m pip install -r requirements.txt

run: venv/bin/activate
	./venv/bin/python3 main.py

install_data:
	mkdir data
	wget -O data/squad_train.json https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
	wget -O data/squad_dev.json https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json
	wget -O data/glove.6B.zip https://nlp.stanford.edu/data/glove.6B.zip && unzip data/glove.6B.zip -d data/
	rm data/glove.6B.zip

	{PYTHON} -m venv venv
	./venv/bin/python -m pip install -r requirements.txt