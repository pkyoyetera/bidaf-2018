import json

from collections import Counter
from pathlib import Path
from typing import Callable, List

from .utils import normalize_answer


def max_over_ground_truths(metric_fn: Callable[[str, List], float],
                           prediction: str,
                           ground_truths: List):
    """
    Returns maximum value of metrics for model prediction multiple ground truths.

    :param func metric_fn: 'exact_match_score' or 'f1_score' functions
    :param str prediction: predicted answer span
    :param list ground_truths: list of ground truths
    """
    gt_scores = []
    for gt in ground_truths:
        score = metric_fn(prediction, gt)
        gt_scores.append(score)

    return max(gt_scores)


def evaluate(predictions):
    """
    Take predictions dictionary where the query_id is the key and the prediction is the value.
    Compare to actual values from the original validation set and take the answer that gives the
    max value for exact match or F1 score.
    :param predictions: dict of model predictions.
    :return: em: 1 if prediction matches the ground truth exactly, 0 otherwise
    """
    parent_path = Path(".").parent.absolute().as_posix()
    with open(parent_path+'/data/squad_dev.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    data = data['data']
    f1 = 0.
    em = 0
    total = 0

    for example in data:
        for paragraph in example['paragraphs']:
            for qa in paragraph['qas']:
                # total += 1
                if qa['id'] not in predictions:
                    continue

                total += 1

                ground_truths = list(map(lambda x: x['text'], qa['answers']))

                pred = predictions[qa['id']]

                em += max_over_ground_truths(exact_match_score, pred, ground_truths)
                f1 += max_over_ground_truths(f1_score, pred, ground_truths)

    em = 100.0 * em / total
    f1 = 100.0 * f1 / total

    return em, f1


def f1_score(pred, truth):
    """
        F1 score between a predicted string and a ground truth string
    """
    pred_tokens = normalize_answer(pred).split()
    truth_tokens = normalize_answer(truth).split()
    common = Counter(pred_tokens) & Counter(truth_tokens)

    intersection = sum(common.values())
    if intersection == 0:
        return 0

    precision = 1.0 * intersection / len(pred_tokens)
    recall = 1.0 * intersection / len(truth_tokens)

    return (2*precision*recall) / (precision + recall)


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)
