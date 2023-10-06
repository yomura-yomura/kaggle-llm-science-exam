import torch


def get_model_kwargs(quant_n_bits: int | None) -> dict:
    model_kwargs = dict(
        trust_remote_code=True,
        device_map="auto",
    )
    if quant_n_bits is None:
        pass
    elif quant_n_bits == 4:
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        model_kwargs["quantization_config"] = bnb_config
    elif quant_n_bits == 16:
        model_kwargs["torch_dtype"] = torch.float16
    elif quant_n_bits == 32:
        model_kwargs["torch_dtype"] = torch.float32
    else:
        raise ValueError(f"unexpected quant_n_bits: {quant_n_bits}")
    return model_kwargs
