import torch
from torch import nn


class Perplexity(nn.Module):
    def __init__(self, reduce: bool = True):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.reduce = reduce

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        perplexity = torch.stack(
            [self.loss_fn(shift_logits[i], shift_labels[i]) for i in range(labels.shape[0])], dim=0
        )
        if self.reduce:
            perplexity = torch.mean(perplexity)
        return perplexity
