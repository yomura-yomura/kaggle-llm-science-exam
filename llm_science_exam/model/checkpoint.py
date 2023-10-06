import json
import pathlib
import shutil

from ..typing import FilePath


def get_latest_checkpoint_path(ckpt_path: FilePath) -> pathlib.Path:
    ckpt_path = pathlib.Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"given ckpt_path not exists: {ckpt_path}")

    ckpt_paths = sorted(ckpt_path.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    latest_ckpt_path = ckpt_paths[-1]
    print(f"latest: {latest_ckpt_path}")
    return latest_ckpt_path


def get_best_checkpoint_path(ckpt_path: FilePath, symlink_to_best: bool = True):
    with open(get_latest_checkpoint_path(ckpt_path) / "trainer_state.json") as f:
        trainer_state = json.load(f)

    ckpt_path = pathlib.Path(ckpt_path)
    best_ckpt_path = pathlib.Path(trainer_state["best_model_checkpoint"])
    best_ckpt_path = ckpt_path / best_ckpt_path.name
    print(f"best: {best_ckpt_path}")
    if not best_ckpt_path.exists():
        raise FileNotFoundError(best_ckpt_path)

    shutil.copy(ckpt_path / "train_config.json", best_ckpt_path / "train_config.json")
    if symlink_to_best:
        best_symlink_ckpt_path = ckpt_path / "best"
        best_symlink_ckpt_path.unlink(missing_ok=True)
        best_symlink_ckpt_path.symlink_to(best_ckpt_path.name)
        return best_symlink_ckpt_path
    else:
        return best_ckpt_path
