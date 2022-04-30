import torch.nn.functional as F


def train(model, trainset, optim, device='cpu'):
    """ Train function """
    loss = 0.
    batch_ct = 0

    # set model to train phase
    model.train()

    for batch in trainset:
        optim.zero_grad()
        if batch_ct % 20 == 0:
            print(f"\rBatch: {batch_ct} of {len(trainset)}", end='', flush=True)
        batch_ct += 1

        context, query, context_chars, query_chars, label, context_emb, ans, ids = batch

        # Move to GPU if specified, otherwise use default (cpu)
        context = context.to(device)
        query = query.to(device)
        context_chars = context_chars.to(device)
        query_chars = query_chars.to(device)
        label = label.to(device)

        # get model predictions
        start_prediction, end_prediction = model(context, query, context_chars, query_chars)

        # get correct labels
        start_index, end_index = label[:, 0], label[:, 1]

        # Compute loss
        _loss = F.cross_entropy(start_prediction, start_index) +\
                F.cross_entropy(end_prediction, end_index)
        _loss.backward()

        optim.step()

        loss += _loss.item()

    print("")
    return loss / len(trainset)
