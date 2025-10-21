import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from huggingface_sb3 import package_to_hub

repo_id = "paudelsamir/ppo-LunarLander-sam"
env_id = "LunarLander-v3"
model_name = "ppo-LunarLander-sam"
model_architecture = "PPO"
commit_message = "Initial commit with trained PPO model on LunarLander-v3"

model = PPO.load(f"{model_name}.zip")

eval_env = DummyVecEnv([lambda: gym.make(env_id, render_mode="rgb_array")])

package_to_hub(
    model=model,
    model_name=model_name,
    model_architecture=model_architecture,
    env_id=env_id,
    eval_env=eval_env,
    repo_id=repo_id,
    commit_message=commit_message
)