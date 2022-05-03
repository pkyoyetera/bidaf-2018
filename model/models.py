import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharacterEmbeddingLayer(nn.Module):
    def __init__(self, character_vocab_dim, char_embedding_dim, num_output_channels, kernel_size):

        super().__init__()
        self.char_emb_dim = char_embedding_dim
        self.char_embedding = nn.Embedding(character_vocab_dim,
                                           char_embedding_dim,
                                           padding_idx=1)
        self.character_conv = nn.Conv2d(in_channels=1,
                                        out_channels=100,
                                        kernel_size=kernel_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        """
        Takes an input batch and returns an embedding of dimension 100
        :param x: input with shape [batch_size, sequence_length, word_length]
        :return:
        """
        batch_size = x.shape[0]

        x = self.dropout(self.char_embedding(x))  # x now has shape [batch_size, sequence_length,
        x = x.permute(0, 1, 3, 2)
        x = x.view(-1, self.char_emb_dim, x.shape[3])

        x = x.unsqueeze(1)

        x = self.relu(self.character_conv(x)).squeeze()

        x = F.max_pool1d(x, x.shape[2]).squeeze()

        x = x.view(batch_size, -1, x.shape[1])

        return x


class HighwayNetwork(nn.Module):
    """ Highway Network. """
    def __init__(self, in_dimension, layers=2):
        super(HighwayNetwork, self).__init__()

        self.num_layers = layers

        self.flow = nn.ModuleList(
            [nn.Linear(in_dimension, in_dimension) for _ in range(self.num_layers)]
        )
        self.gate = nn.ModuleList(
            [nn.Linear(in_dimension, in_dimension) for _ in range(self.num_layers)]
        )

    def forward(self, x):
        for i in range(self.num_layers):
            flow_value = F.relu(self.flow[i](x))
            gate_value = torch.sigmoid((self.gate[i](x)))

            x = gate_value * flow_value + (1 - gate_value) * x

        return x


class ContextualEmbedding(torch.nn.Module):
    """
        Contextual Embedding layer
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            batch_first=True,
                            bidirectional=True)
        self.highway_layer = HighwayNetwork(input_dim)

    def forward(self, x):
        out = self.highway_layer(x)

        outs, _ = self.lstm(out)

        return outs


class BiDirectionalAttentionFlow(torch.nn.Module):

    def __init__(self,
                 c_vocab_size,
                 embedding_dim,
                 c_embedding_dim,
                 out_channels,
                 kernel_size,
                 ctx_hidden_dim,
                 device):

        super().__init__()

        self.device = device
        self.word_embeds = self.get_glove_embeds()

        self.char_embeds = CharacterEmbeddingLayer(c_vocab_size,
                                                   c_embedding_dim,
                                                   out_channels,
                                                   kernel_size)

        self.context_embed = ContextualEmbedding(embedding_dim*2, ctx_hidden_dim)

        self.dropout = torch.nn.Dropout()

        self.similarity_weight = nn.Linear(embedding_dim*6, 1, bias=False)

        self.modeling_lstm = nn.LSTM(embedding_dim*8,
                                     embedding_dim,
                                     bidirectional=True,
                                     num_layers=2,
                                     batch_first=True,
                                     dropout=0.2)

        self.output_begin = nn.Linear(embedding_dim*10, 1, bias=False)
        self.output_end = nn.Linear(embedding_dim*10, 1, bias=False)

        self.end_lstm = nn.LSTM(embedding_dim*2,
                                embedding_dim,
                                bidirectional=True,
                                batch_first=True)

    def forward(self, context, question, char_context, char_question):
        context_length = context.shape[1]
        question_length = question.shape[1]

        context_word_embedding = self.word_embeds(context)
        question_word_embedding = self.word_embeds(question)

        context_char_embedding = self.char_embeds(char_context)
        question_char_embedding = self.char_embeds(char_question)

        # Contextual embedding is a concatenation of the context word and character representation/embedding
        context_input = torch.cat([context_word_embedding, context_char_embedding], dim=2)
        question_input = torch.cat([question_word_embedding, question_char_embedding], dim=2)

        contextual_embedding_ctx = self.context_embed(context_input)
        contextual_embedding_question = self.context_embed(question_input)

        # Similarity matrix
        ctx = contextual_embedding_ctx.unsqueeze(2).repeat(1, 1, question_length, 1)
        qun = contextual_embedding_question.unsqueeze(1).repeat(1, context_length, 1, 1)

        prod = torch.mul(ctx, qun)
        alpha = torch.cat([ctx, qun, prod], dim=3)
        similarity_mat = self.similarity_weight(alpha).view(-1, context_length, question_length)

        # Context-to-Query Attention
        att_q = F.softmax(similarity_mat, dim=-1)
        context_to_query = torch.bmm(att_q, contextual_embedding_question)

        # Query-to-Context Attention
        att_c = F.softmax(torch.max(similarity_mat, 2)[0], dim=-1).unsqueeze(1)
        query_to_context = torch.bmm(att_c, contextual_embedding_ctx)
        query_to_context = query_to_context.repeat(1, context_length, 1)

        # The Query-aware representation
        G = torch.cat([contextual_embedding_ctx,
                       context_to_query,
                       torch.mul(contextual_embedding_ctx, context_to_query),
                       torch.mul(contextual_embedding_ctx, query_to_context)],
                      dim=2)

        # The modeling layer
        M, _ = self.modeling_lstm(G)

        output, _ = self.end_lstm(M)

        # Prediction for answer start
        begin_pred = self.output_begin(torch.cat([G, M], dim=2)).squeeze()

        # Prediction for answer end
        end_pred = self.output_end(torch.cat([G, output], dim=2)).squeeze()

        return begin_pred, end_pred

    def get_glove_embeds(self):
        weights_matrix = np.load('data/bidafglove_short_tv.npy')

        # embeds_ct, embed_dim = weights_matrix.shape
        embed = nn.Embedding.from_pretrained(torch.FloatTensor(weights_matrix).to(self.device), freeze=True)

        return embed

