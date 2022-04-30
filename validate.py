import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.evaluate import evaluate


# Validation function
def validate(model, validation_set, index2word, device='cpu'):
    loss = 0.
    ct = 0

    # Set model into evaluation phase
    model.eval()

    predictions = {}

    for batch in validation_set:
        if ct % 10 == 0:
            print(f"\rValidation batch: {ct} of {len(validation_set)}", end='', flush=True)
        ct += 1

        context, query, context_chars, query_chars, label, context_emb, ans, ids = batch

        # Move to GPU if specified, otherwise use default (cpu)
        context = context.to(device)
        query = query.to(device)
        context_chars = context_chars.to(device)
        query_chars = query_chars.to(device)
        label = label.to(device)

        with torch.no_grad():
            start_index, end_index = label[:, 0], label[:, 1]

            # Get model predictions
            p1, p2 = model(context, query, context_chars, query_chars)

            _loss = F.cross_entropy(p1, start_index) + F.cross_entropy(p2, end_index)
            loss += _loss.item()

            batch_size, c_len = p1.size()

            softmax = nn.LogSoftmax(dim=1)
            mask = (torch.ones(c_len, c_len) * float('-inf')).to(device).tril(-1).unsqueeze(0).expand(batch_size, -1, -1)
            score = (softmax(p1).unsqueeze(2) + softmax(p2).unsqueeze(1)) + mask
            score, start_index = score.max(dim=1)
            score, end_index = score.max(dim=1)
            start_index = torch.gather(start_index, 1, end_index.view(-1, 1)).squeeze()

            for i in range(batch_size):
                _id = ids[i]

                pred = context[i][start_index[i]: end_index[i]+1]
                pred = ' '.join([index2word[i.item()] for i in pred])

                predictions[_id] = pred

    # F1 and exact match scores
    em, f1 = evaluate(predictions)

    print("")

    return loss / len(validation_set), em, f1
