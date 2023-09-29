import llm_science_exam.data
import llm_science_exam.llama2.model

# config_path = "config/llama2.toml"
# config_path = "config/platypus2.toml"
# config_path = "config/openorca-platypus2.toml"
# config_path = "config/v100/llama2.toml"
# config_path = "config/llama2-with-context-500w.toml"
# config_path = "config/llama2-with-context-300w.toml"
# config_path = "config/a100/llama2-with-context-300w.toml"
config_path = "config/a100/llama2-16bits.toml"


config = llm_science_exam.data.config.get_config(config_path)
ckpt_path = llm_science_exam.data.config.get_checkpoint_path(config)

ckpt_path = llm_science_exam.llama2.model.get_latest_checkpoint_path(ckpt_path)


# ckpt_path = "models/llama2/05-with-context/7b/300w-4/_checkpoint-8300/"

fig = llm_science_exam.llama2.plot.get_learning_curve_plot(ckpt_path)
# fig.update_xaxes(range=[0.1, None])
# fig.update_yaxes(range=[0.85, 1], row=1)
fig.show()
