"""Metrics computation utilities.

Adapted from SimCLR PyTorch implementation
Copyright (c) 2020 Thalles Silva
Repository: https://github.com/sthalles/SimCLR
Commit: 1848fc934ad844ae630e6c452300433fe99acfd9
MIT License
"""

import torch


def accuracy(output, target, topk=(1,)):
    """Compute the accuracy over the k top predictions for the specified values of k.

    Args:
        output: Model predictions of shape (batch_size, num_classes)
        target: Ground truth labels of shape (batch_size,)
        topk: Tuple of k values for top-k accuracy

    Returns:
        list: List of top-k accuracies (as percentages)
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
