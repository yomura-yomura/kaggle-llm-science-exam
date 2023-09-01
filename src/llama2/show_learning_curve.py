import json
import pathlib

import numpy as np
import pandas as pd
import plotly.express as px

# ckpt_path = pathlib.Path("SFT-llama2-7b") / "02-prompt-fixed" / "1000steps"
# ckpt_path = pathlib.Path("SFT-llama2-7b") / "with_early_stopping_with_more_datasets"
# ckpt_path = pathlib.Path("SFT-llama2-7b") / "1000steps_with_more_datasets"
# ckpt_path = pathlib.Path("SFT-llama2-7b") / "with_early_stopping_with_2_more_datasets"

# ckpt_path = pathlib.Path("SFT-llama2-13b") / "with_early_stopping_with_2_more_datasets"
# ckpt_path = pathlib.Path("SFT-llama2-13b") / "with_early_stopping_with_2_more_datasets_tol50" / "checkpoint-5800"

# ckpt_path = (
#     pathlib.Path("models") / "SFT-llama2-7b" / "with_early_stopping_with_2_more_datasets_tol50" / "checkpoint-2200"
# )

ckpt_path = pathlib.Path("models/SFT-llama2-7b/with_extra_+6.5k_+13.5k_+1k/checkpoint-7900")


with open(ckpt_path / "trainer_state.json") as f:
    trainer_state = json.load(f)

df = pd.DataFrame(trainer_state["log_history"])

if "eval_loss" not in df.columns:
    df["eval_loss"] = np.nan


n_steps_at_best = int(trainer_state["best_model_checkpoint"].rsplit("-", maxsplit=1)[1])
n_epochs_at_best = n_steps_at_best * trainer_state["epoch"] / trainer_state["global_step"]

lc_df = df.melt(id_vars="epoch", value_vars=["loss", "eval_loss"]).rename(columns=dict(variable="type", value="loss"))
fig = px.line(lc_df, title="Learning Curve", x="epoch", y="loss", color="type")
fig.update_traces(connectgaps=True)
fig.add_vline(x=n_epochs_at_best, line_dash="dash", annotation_text=f"Saved at loss={trainer_state['best_metric']:.3f}")
fig.show()
