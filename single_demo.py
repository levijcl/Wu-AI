import gymnasium as gym
import numpy as np
import time
from dragon_gate_env import DragonGateEnv
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback

# Custom callback to render the environment during training
class RenderCallback(BaseCallback):
    def __init__(self, render_freq=100, verbose=0):
        super(RenderCallback, self).__init__(verbose)
        self.render_freq = render_freq  # Render every N steps
        self.step_count = 0

    def _on_step(self) -> bool:
        self.step_count += 1
        if self.step_count % self.render_freq == 0:
            self.training_env.env_method("render")
            # Add a short sleep to make rendering visible
            time.sleep(0.05)
        return True

# Register the custom environment with gymnasium
gym.register(
    id='DragonGate-v0',
    entry_point='dragon_gate_env:DragonGateEnv',
)

# Try to use graphical rendering with fallback to text-based if needed
try:
    env = gym.make('DragonGate-v0', render_mode="ansi", num_players=4, min_bet=100,
                   starting_money=1000, max_rounds=100)
    print("Using graphical rendering")
except:
    env = gym.make('DragonGate-v0', render_mode="ansi", num_players=4, min_bet=100,
                   starting_money=1000, max_rounds=100)
    print("Using text-based rendering")

# Check observation space compatibility
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")

# Configure the RL model with appropriate policy for dictionary observation space
model = A2C("MultiInputPolicy", env, verbose=1)

# Create the render callback
render_callback = RenderCallback(render_freq=100)  # Render every 100 steps during training

# Train the model with the callback to show the environment during training
print("Starting training with visualization every 100 steps...")
model.learn(total_timesteps=10_000, callback=render_callback)
print("Training completed!")

# Test the trained model
print("\nTesting the trained model:")
obs, _ = env.reset()
for i in range(30):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()

    print(f"Action: bet={int(action[0]*100)}%, choice={'Higher' if action[1] > 0.5 else 'Lower'}")

    if terminated or truncated:
        print(f"Game Over! Final money: {info['money']}")
        break

env.close()
