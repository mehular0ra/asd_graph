import torch


def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """Computes the top-1 accuracy"""
    _, predicted = torch.max(output, 1)
    correct = (predicted == target).float()
    accuracy = correct.sum() * 100 / target.size(0)
    return accuracy.item()


def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False
