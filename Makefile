
venv/bin/activate: requirements.txt
	python3.8 -m venv venv
	./venv/bin/activate

run: venv/bin/activate
	venv/bin/python3 main.py

install_data:
	wget -O data/squad_train.json https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
	wget -O data/squad_dev.json https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json
	wget -O data/glove.6B.zip https://nlp.stanford.edu/data/glove.6B.zip && unzip data/glove.6B.zip -d data/
	rm data/data/glove.6B.zip