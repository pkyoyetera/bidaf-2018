import torch
import spacy
# import en_core_web_sm

# nlp = en_core_web_sm.load()


def get_span(text):
    text = SquadDataset.nlp(text, disable=['parser', 'tagger', 'ner'])
    span = [(w.idx, w.idx+len(w.text)) for w in text]

    return span


class SquadDataset:
    nlp = spacy.load("en_core_web_sm")

    def __init__(self, data, batch_size, char2idx):

        self.batch_size = batch_size
        data = [data[i: i +self.batch_size] for i in range(0, len(data), self.batch_size)]
        self.data = data
        self.char2idx = char2idx

    def __len__(self):
        return len(self.data)

    def make_char_vector(self, max_sent_len, max_word_len, sentence):

        char_vec = torch.ones(max_sent_len, max_word_len).type(torch.LongTensor)

        for i, word in enumerate(SquadDataset.nlp(sentence, disable=['parser' ,'tagger' ,'ner'])):
            for j, ch in enumerate(word.text):
                char_vec[i][j] = self.char2idx.get(ch, 0)

        return char_vec

    def __iter__(self):
        for batch in self.data:
            spans = []
            ctx_text = []
            answer_text = []

            for ctx in batch.context:
                ctx_text.append(ctx)
                spans.append(get_span(ctx))

            for ans in batch.answer:
                answer_text.append(ans)

            max_context_len = max([len(ctx) for ctx in batch.context_ids])
            padded_context = torch.LongTensor(len(batch), max_context_len).fill_(1)

            for i, ctx in enumerate(batch.context_ids):
                padded_context[i, :len(ctx)] = torch.LongTensor(ctx)

            max_word_ctx = 0
            for context in batch.context:
                for word in SquadDataset.nlp(context, disable=['parser' ,'tagger' ,'ner']):
                    if len(word.text) > max_word_ctx:
                        max_word_ctx = len(word.text)

            char_ctx = torch.ones(len(batch), max_context_len, max_word_ctx).type(torch.LongTensor)
            for i, context in enumerate(batch.context):
                char_ctx[i] = self.make_char_vector(max_context_len, max_word_ctx, context)

            max_question_len = max([len(ques) for ques in batch.question_ids])
            padded_question = torch.LongTensor(len(batch), max_question_len).fill_(1)

            for i, ques in enumerate(batch.question_ids):
                padded_question[i, :len(ques)] = torch.LongTensor(ques)

            max_word_ques = 0
            for question in batch.question:
                for word in SquadDataset.nlp(question, disable=['parser', 'tagger', 'ner']):
                    if len(word.text) > max_word_ques:
                        max_word_ques = len(word.text)

            char_ques = torch.ones(len(batch), max_question_len, max_word_ques).type(torch.LongTensor)
            for i, question in enumerate(batch.question):
                char_ques[i] = self.make_char_vector(max_question_len, max_word_ques, question)

            ids = list(batch.id)
            label = torch.LongTensor(list(batch.label_idx))

            yield padded_context, padded_question, char_ctx, char_ques, label, ctx_text, answer_text, ids
