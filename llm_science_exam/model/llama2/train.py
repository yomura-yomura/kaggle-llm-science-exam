import os
import pathlib
import shutil

from transformers import TrainingArguments
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from ..train import get_training_args

__all__ = ["get_training_args", "SavePeftModelCallback"]


class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = pathlib.Path(args.output_dir) / f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"

        kwargs["model"].save_pretrained(checkpoint_folder)

        pytorch_model_path = checkpoint_folder / "pytorch_model.bin"
        if pytorch_model_path.exists():
            os.remove(pytorch_model_path)

        train_config_json = checkpoint_folder.parent / "train_config.json"
        if train_config_json.exists():
            shutil.copy(train_config_json, checkpoint_folder / train_config_json.name)

        return control
