import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from ..typing import NDArray


def get_embeddings(
    model: SentenceTransformer, sentences: list[str], *, device: str | None = None, batch_size: int = 16
) -> NDArray[np.float_]:
    with torch.no_grad():
        embeddings = model.encode(
            sentences,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_tensor=True,
            normalize_embeddings=True,
            device=device or model.device,
        )
    return embeddings.detach().cpu().numpy()
