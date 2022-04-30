import json
import nltk
import re
import spacy
import string
import unicodedata

import numpy as np
import pandas as pd

from collections import Counter
from typing import Dict, List, Set


def parse_data(data: Dict) -> List:
    """ Parse a JSON SQuAD dataset """
    data = data['data']
    qa_list = []

    for paragraphs in data:
        for p in paragraphs['paragraphs']:
            context = p['context']

            for qa in p['qas']:
                _id = qa['id']
                question = qa['question']

                for ans in qa['answers']:
                    answer = ans['text']
                    start = ans['answer_start']
                    end = start + len(answer)

                    qa_list.append({'id': _id,
                                    'context': context,
                                    'question': question,
                                    'label': [start, end],
                                    'answer': answer})

    return qa_list


def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data


nlp = spacy.load('en_core_web_sm')


def filter_large(example: pd.DataFrame):
    """
    Some context, questions and/or answers may be too large to process,
    we need to remove those from the dataset
    :param example: pandas dataframe
    :return: set
    """
    context_length = []
    query_length = []
    answer_length = []

    for i, row in example.iterrows():
        # Check number of tokens in context
        context_tokens = [word.text for word in nlp(row.context, disable=['parser', 'ner', 'tagger'])]
        if len(context_tokens) > 400:
            context_length.append(row.name)

        # Check number of tokens in query
        query_tokens = [word.text for word in nlp(row.context, disable=['parser', 'ner', 'tagger'])]
        if len(query_tokens) > 50:
            query_length.append(row.name)

        # Check number of tokens in answer
        answer_tokens = [word.text for word in nlp(row.context, disable=['parser', 'ner', 'tagger'])]
        if len(answer_tokens) > 30:
            answer_length.append(row.name)

        assert row.name == i

    return set(answer_length + context_length + query_length)


# Tokenize and normalize quotes
def word_tokenizer(text: str) -> List[str]:
    # fixme: deal with double and single quotes elsewhere as they mess with indices
    # return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(text)]
    return nltk.word_tokenize(text.strip())

# First define all the letters we're to expect including padding, unknown,
# start of sentence and end of sentence tokens as letter to expect
all_letters = string.ascii_letters + " .,;'-"
num_letters = len(all_letters) + 4


# Function to convert unicode letters to ascii letters
def unicode_to_ascii(text: str) -> str:
    return ''.join (c for c in unicodedata.normalize('NFD', text)
                    if unicodedata.category(c) != 'Mn' and c in all_letters)


# Lowercase each token
def to_lower(tokens: List[str]) -> List[str]:
    return [token.lower() for token in tokens]


def build_word_vocab(text_set: Set):
    """
    Build a word-level vocabulary from the given text
    :param text_set: set of lower-cased, tokenized strings from all the training/val sets
    :return: vocabulary
    """
    # _vocab = torchtext.vocab.build_vocab_from_iterator(text_set, min_freq=0, specials=['<unk>', '<pad>'])
    # _vocab.set_default_index(_vocab['<unk>'])
    # return _vocab

    words = []
    for sent in text_set:
        for word in nlp(sent, disable=['parser', 'ner', 'tagger']):
            words.append(word.text)

    counter = Counter(words)
    word_vocabulary = sorted(counter, key=counter.get, reverse=True)
    word_vocabulary.insert(0, '<unk>')
    word_vocabulary.insert(1, '<pad>')

    w2i = {word: index for index, word in enumerate(word_vocabulary)}
    i2w = {index: word for word, index in w2i.items()}

    return w2i, i2w, word_vocabulary


def build_character_vocabulary(text_set: Set[str]):
    """
    Build a vocabulary of the characters in the text
    :param text_set: set of tokens that appear in the text
    :return: a dictionary of characters
    """
    chars = [ch for token in text_set for ch in token]

    # keep track of counts for characters in text
    character_counts = Counter(chars)

    # sort characters by frequency
    char_vocab = sorted(character_counts, key=character_counts.get, reverse=True)

    print(f'Raw character counts: {len(char_vocab)}')

    # which characters occur the most?
    # most_frequent = [c for c, count in character_counts.items() if count > 20]

    char_vocab.insert(0, '<unk>')
    char_vocab.insert(1, '<pad>')

    # Create our character to index mapping
    char2idx = {c: idx for idx, c in enumerate(char_vocab)}
    print(f'Size of char2idx: {len(char2idx)}')

    return char2idx, char_vocab


def collect_text_for_vocabulary(train_df: pd.DataFrame, val_df: pd.DataFrame) -> List[str]:
    """
        Vocabulary from text.
        Create a vocabulary from all the available text
        :param train_df Dataframe with train set
        :param val_df: validation dataframe
    """
    # gather text from train dataframe
    _total_text = []
    _total_text += list(train_df.context.unique())
    _total_text += list(val_df.context.unique())
    _total_text += list(train_df.question.unique())
    _total_text += list(val_df.question.unique())

    return _total_text


def normalize_answer(s):
    """
    Perform a series of cleaning steps on the ground truth and predicted answer.
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def preprocess_dataframe(df: pd.DataFrame):
    def _to_lower(text: str) -> str:
        return text.lower()

    df.context = df.context.apply(_to_lower)
    df.question = df.question.apply(_to_lower)
    df.answer = df.answer.apply(_to_lower)


def context_to_ids(text: str, word2idx: Dict):
    context_tokens = [w.text for w in nlp(text, disable=['parser', 'tagger', 'ner'])]
    context_ids = [word2idx[word] for word in context_tokens]

    assert len(context_ids) == len(context_tokens)
    return context_ids


def question_to_ids(text: str, word2idx: Dict):
    question_tokens = [w.text for w in nlp(text, disable=['parser', 'tagger', 'ner'])]
    question_ids = [word2idx[word] for word in question_tokens]

    assert len(question_ids) == len(question_tokens)
    return question_ids


def test_indices(df, idx2word):
    start_value_error = []
    end_value_error = []
    assert_error = []
    for index, row in df.iterrows():

        answer_tokens = [w.text for w in nlp(row['answer'], disable=['parser', 'tagger', 'ner'])]

        start_token = answer_tokens[0]
        end_token = answer_tokens[-1]

        context_span = [(word.idx, word.idx + len(word.text))
                        for word in nlp(row['context'], disable=['parser', 'tagger', 'ner'])]

        starts, ends = zip(*context_span)

        answer_start, answer_end = row['label']

        try:
            start_idx = starts.index(answer_start)
        except Exception as e:
            start_value_error.append(index)

        try:
            end_idx = ends.index(answer_end)
        except Exception as e:
            end_value_error.append(index)

        try:
            assert idx2word[row['context_ids'][start_idx]] == answer_tokens[0]
            assert idx2word[row['context_ids'][end_idx]] == answer_tokens[-1]
        except:
            assert_error.append(index)

    return start_value_error, end_value_error, assert_error


def get_error_indices(df, idx2word):
    start_value_error, end_value_error, assert_error = test_indices(df, idx2word)
    err_idx = start_value_error + end_value_error + assert_error
    err_idx = set(err_idx)
    print(f"Number of error indices: {len(err_idx)}")

    return err_idx


def answer_index(row, idx2word):
    """
    Takes in a row of the dataframe or one training example and
    returns a tuple of start and end positions of answer by calculating
    spans.
    """
    context_span = [(word.idx, word.idx + len(word.text))
                    for word in nlp(row.context, disable=['parser', 'tagger', 'ner'])]
    starts, ends = zip(*context_span)

    answer_start, answer_end = row.label
    start_idx = starts.index(answer_start)

    end_idx = ends.index(answer_end)

    ans_toks = [w.text for w in nlp(row.answer, disable=['parser', 'tagger', 'ner'])]
    ans_start = ans_toks[0]
    ans_end = ans_toks[-1]
    assert idx2word[row.context_ids[start_idx]] == ans_start
    assert idx2word[row.context_ids[end_idx]] == ans_end

    return [start_idx, end_idx]


def get_glove_dict():
    glove_dict = {}
    with open("../data/glove.6B.100d.txt", "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            try:
                vector = np.asarray(values[1:], "float32")
                glove_dict[word] = vector
            except ValueError as ve:
                ...

    f.close()

    return glove_dict


def create_weights_matrix(glove_dict, word_vocab):
    weights_matrix = np.zeros((len(word_vocab), 100))
    words_found = 0
    for i, word in enumerate(word_vocab):
        try:
            weights_matrix[i] = glove_dict[word]
            words_found += 1
        except:
            pass

    return weights_matrix, words_found
