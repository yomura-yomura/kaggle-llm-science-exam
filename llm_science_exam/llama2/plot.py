import json
import pathlib
from typing import Literal

import numpy as np
import pandas as pd
import plotly.express as px
import torch

from ..typing import FilePath


def get_learning_curve_plot(ckpt_path: FilePath, save_point: Literal["best", "latest"] = "best"):
    ckpt_path = pathlib.Path(ckpt_path)
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

    other_metrics = df.columns[
        df.columns.str.startswith("eval_")
        & ~df.columns.isin(["eval_loss", "eval_runtime", "eval_samples_per_second", "eval_steps_per_second"])
    ].tolist()

    lc_df = df.melt(
        id_vars=["epoch", "step"],
        value_vars=[
            "loss",
            "eval_loss",
            *other_metrics,
        ],
    ).rename(columns=dict(variable="type", value="value"))

    latest_eval_loss = (
        lc_df.dropna().query("type == 'eval_loss'").query(f"step == {n_steps_at_latest}")["value"].iloc[0]
    )

    lc_df["col"] = "loss"
    for m in other_metrics:
        lc_df.loc[lc_df["type"] == m, "col"] = m.replace("eval_", "")

    fig = px.line(
        lc_df,
        title=f"Learning Curve (lr={training_args['learning_rate']}, bs/dev={training_args['per_device_train_batch_size']})",
        x="epoch",
        y="value",
        facet_row="col",
        color="type",
        category_orders={"col": ["loss"]},
    )
    fig.update_traces(connectgaps=True)
    fig.update_yaxes(matches=None)
    fig.update_yaxes(
        range=[None, lc_df.query("type.isin(['loss', 'eval_loss'])").iloc[:10]["value"].max() * 1.1], row=-1
    )

    if save_point == "best":
        fig.add_vline(
            x=n_epochs_at_best,
            line_dash="dash",
            annotation_text=f"Saved at y={trainer_state['best_metric']:.3f}",
            row=1,
        )
    elif save_point == "latest":
        fig.add_vline(
            x=n_epochs_at_latest, line_dash="dash", annotation_text=f"Saved at y={latest_eval_loss:.3f}", row=1
        )
    else:
        raise ValueError(f"unexpected save_point: {save_point}")
    return fig
