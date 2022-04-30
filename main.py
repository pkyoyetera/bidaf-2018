import itertools
import os.path
import pickle

import numpy as np
import pandas as pd

import torch
import torch.optim as optim

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# local imports
from dataset import SquadDataset
from model.models import BiDirectionalAttentionFlow
from train import train
from utils.utils import (answer_index,
                         build_character_vocabulary,
                         build_word_vocab,
                         collect_text_for_vocabulary,
                         context_to_ids,
                         create_weights_matrix,
                         filter_large,
                         get_error_indices,
                         get_glove_dict,
                         load_json,
                         parse_data,
                         preprocess_dataframe,
                         question_to_ids)
from validate import validate

EPOCHS = 5

EMB_DIM = 100
# EMB_DIM = 300
CHAR_EMB_DIM = 8
NUM_OUTPUT_CHANNELS = 100
KERNEL_SIZE = (8, 4)
HIDDEN_DIM = 100
BATCH_SIZE = 4


def process_data():
    train_data = load_json('data/squad_train.json')
    valid_data = load_json('data/squad_dev.json')

    train_list = parse_data(train_data)
    dev_list = parse_data(valid_data)

    tr_thirds = len(train_list) // 3
    dev_third = len(dev_list) // 3

    train_list = list(itertools.islice(train_list, tr_thirds))
    dev_list = list(itertools.islice(dev_list, dev_third))

    train_df = pd.DataFrame(train_list)
    val_df = pd.DataFrame(dev_list)

    return train_df, val_df


def main():
    # Check if processed data exists. Create new data if it does not
    if os.path.isfile('data/bidaf_train.pkl') and \
            os.path.isfile('data/bidaf_valid.pkl') and \
            os.path.isfile('data/bidafw2id.pickle') and \
            os.path.isfile('data/bidafc2id.pickle') and \
            os.path.isfile('data/bidafglove_short_tv.npy'):

        # Data exists, load it
        print("Loading data...")
        start = datetime.now()
        train_set = pd.read_pickle('data/bidaf_train.pkl')
        dev_set = pd.read_pickle('data/bidaf_valid.pkl')

        with open('data/bidafw2id.pickle', 'rb') as handle:
            word2index = pickle.load(handle)
        with open('data/bidafc2id.pickle', 'rb') as handle:
            char2index = pickle.load(handle)

        index2word = {v: k for k, v in word2index.items()}
        end = datetime.now()
        print(f"Done loading data. Time: {end-start}")

    else:
        train_set, dev_set = process_data()

        # Preprocess data in dataframes
        preprocess_dataframe(train_set)
        preprocess_dataframe(dev_set)

        # Create a vocabulary from all the text in train and test sets
        print("Creating vocabulary...")
        start = datetime.now()
        vocab_text = collect_text_for_vocabulary(train_set, dev_set)

        word2index, index2word, word_vocabulary = build_word_vocab(vocab_text)
        char2index, character_vocab = build_character_vocabulary(vocab_text)

        end = datetime.now()
        print(f"Finished vocabulary. Time: {end - start}s")

        print("\nWrapping up dataset creation...")
        start = datetime.now()
        train_set['context_ids'] = train_set.context.apply(context_to_ids, word2idx=word2index)
        dev_set['context_ids'] = dev_set.context.apply(context_to_ids, word2idx=word2index)

        train_set['question_ids'] = train_set.question.apply(question_to_ids, word2idx=word2index)
        dev_set['question_ids'] = dev_set.question.apply(question_to_ids, word2idx=word2index)

        print("...removing examples with wrong indices...")
        # find and remove erroneous answer indices
        t_error = get_error_indices(train_set, index2word)
        d_error = get_error_indices(dev_set, index2word)

        train_set.drop(t_error, inplace=True)
        dev_set.drop(d_error, inplace=True)

        train_set['label_idx'] = train_set.apply(answer_index, axis=1, idx2word=index2word)
        dev_set['label_idx'] = dev_set.apply(answer_index, axis=1, idx2word=index2word)

        print("\nsaving data...\n")

        train_set.to_pickle('data/bidaf_train.pkl')
        dev_set.to_pickle('data/bidaf_valid.pkl')

        with open('data/bidafw2id.pickle', 'wb') as h:
            pickle.dump(word2index, h)
        with open('data/bidafc2id.pickle', 'wb') as h:
            pickle.dump(char2index, h)

        end = datetime.now()
        print(f"Done! Time: {end - start}s")

        weights_matrix, words_found = create_weights_matrix(get_glove_dict(), word_vocabulary)
        np.save('data/bidafglove_short_tv.npy', weights_matrix)

    # create dataset objects
    train_dataset = SquadDataset(train_set,   # Data
                                 BATCH_SIZE,  # batch size
                                 char2index)  # character vocabulary mapping
    test_dataset = SquadDataset(dev_set,      # data
                                BATCH_SIZE,   # Batch size
                                char2index)   # character vocabulary mapping

    CHAR_VOCAB_DIM = len(char2index)

    # Check cuda
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = BiDirectionalAttentionFlow(CHAR_VOCAB_DIM,
                                       EMB_DIM,
                                       CHAR_EMB_DIM,
                                       NUM_OUTPUT_CHANNELS,
                                       KERNEL_SIZE,
                                       HIDDEN_DIM,
                                       device).to(device)

    optimizer = optim.Adadelta(model.parameters())

    epochs = 5

    train_losses, dev_losses, f1_scores, em_scores = [], [], [], []

    writer = SummaryWriter(comment='Bi_GRU')

    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch}")
        train_loss = train(model, train_dataset, optimizer, device)
        dev_loss, em, f1 = validate(model, test_dataset, index2word, device)

        # Record values for visualization
        writer.add_scalar('train loss', train_loss, epoch)
        writer.add_scalar('dev loss', dev_loss, epoch)

        for name, param in model.named_parameters():
            writer.add_histogram(name, param, epoch)

        train_losses.append(train_loss)
        dev_losses.append(dev_loss)
        f1_scores.append(f1)
        em_scores.append(em)

        print(f"\tEpoch details: \n\ttrain loss: {train_loss}")
        print(f"\t\tvalidation loss: {dev_loss}")
        print(f"\t\tExact match: {em}")
        print(f"\t\tF1: {f1}")

    writer.close()


if __name__ == '__main__':
    main()
