# Bidaf-2018
An implementation if Bi-directional Attention flow from this [paper](https://arxiv.org/abs/1611.01603).

## Run

1. install required data. Run `make install` to download data to the `data` directory.
2. Execute `make run` to create a virtual environment, install required packages then run the program.
Program entry is in the `main.py` file.

### References
1. [Bidirectional Attention Flow for Machine Comprehension](https://arxiv.org/abs/1611.01603)
2. [SQuAD 2.0 dataset](https://arxiv.org/pdf/1606.05250.pdf)
3. https://github.com/allenai/allennlp-models/blob/v2.1.0/allennlp_models/rc/predictors/bidaf.py

Written and tested with python3.8. Can't guarantee proper behavior with other versions of python.
