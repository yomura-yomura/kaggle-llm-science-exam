from sentence_transformers import SentenceTransformer

from .. import pj_struct_paths


def get_model(device: str = "cuda") -> SentenceTransformer:
    model_name = (
        pj_struct_paths.get_data_dir_path()
        / "sentencetransformers-allminilml6v2"
        / "sentence-transformers_all-MiniLM-L6-v2"
    )
    max_seq_length = 384

    model = SentenceTransformer(str(model_name), device="cpu")
    model.max_seq_length = max_seq_length
    # model.half()
    model.to(device)
    return model
