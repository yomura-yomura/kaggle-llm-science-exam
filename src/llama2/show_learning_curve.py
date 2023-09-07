import llm_science_exam.data
import llm_science_exam.llama2.model

# config_path = "config/llama2.toml"
config_path = "config/platypus2.toml"


config = llm_science_exam.data.config.get_config(config_path)
ckpt_path = llm_science_exam.data.config.get_checkpoint_path(config)

ckpt_path = llm_science_exam.llama2.model.get_latest_checkpoint_path(ckpt_path)

fig = llm_science_exam.llama2.plot.get_learning_curve_plot(ckpt_path)
fig.show()
