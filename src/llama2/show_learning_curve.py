import json
import pathlib

import numpy as np
import pandas as pd
import plotly.express as px
import torch

import llm_science_exam.llama2.model

# ckpt_path = pathlib.Path("models") / "03-short-prompt" / "SFT-llama2-7b" / "with_extra_+6.5k_+13.5k_+1k"
# ckpt_path = pathlib.Path("models") / "03-short-prompt" / "SFT-llama2-13b" / "with_extra_+6.5k_+13.5k_+1k"

ckpt_path = pathlib.Path("models") / "03-short-prompt2" / "SFT-llama2-7b" / "with_extra_+6.5k_+13.5k_+1k"


ckpt_path = llm_science_exam.llama2.model.get_latest_checkpoint_path(ckpt_path)

with open(ckpt_path / "trainer_state.json") as f:
    trainer_state = json.load(f)

df = pd.DataFrame(trainer_state["log_history"])

if "eval_loss" not in df.columns:
    df["eval_loss"] = np.nan


training_args = vars(torch.load(ckpt_path / "training_args.bin"))


n_steps_at_best = int(trainer_state["best_model_checkpoint"].rsplit("-", maxsplit=1)[1])
n_epochs_at_best = n_steps_at_best * trainer_state["epoch"] / trainer_state["global_step"]

n_steps_at_latest = int(ckpt_path.name.split("-")[1])
n_epochs_at_latest = n_steps_at_latest * trainer_state["epoch"] / trainer_state["global_step"]

lc_df = df.melt(id_vars=["epoch", "step"], value_vars=["loss", "eval_loss"]).rename(
    columns=dict(variable="type", value="loss")
)

latest_eval_loss = lc_df.dropna().query("type == 'eval_loss'").query(f"step == {n_steps_at_latest}")["loss"].iloc[0]


fig = px.line(
    lc_df,
    title=f"Learning Curve (lr={training_args['learning_rate']}, bs/dev={training_args['per_device_train_batch_size']})",
    x="epoch",
    y="loss",
    color="type",
)
fig.update_traces(connectgaps=True)
fig.update_yaxes(range=[0, lc_df.iloc[:10]["loss"].max() * 1.1])
fig.add_vline(x=n_epochs_at_best, line_dash="dash", annotation_text=f"Saved at loss={trainer_state['best_metric']:.3f}")
# fig.add_vline(x=n_epochs_at_latest, line_dash="dash", annotation_text=f"Saved at loss={latest_eval_loss:.3f}")
fig.show()
