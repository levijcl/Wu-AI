"""
Single Player Training Demo

This script demonstrates how to train models with the redesigned DragonGate environment,
showing different training approaches including:
1. Training against the environment (simple case)
2. Training with simulated opponents
3. Evaluating the trained models
"""

import gymnasium as gym
import numpy as np
import time
import os
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from dragon_gate_env import DragonGateEnv
import matplotlib.pyplot as plt

# Register the custom environment
gym.register(
    id='DragonGate-v0',
    entry_point='dragon_gate_env:DragonGateEnv',
)

def train_single_player(total_timesteps=5000, render_training=False, verbose=1):
    """Train a model to play the game as a single player"""
    print(f"\n{'='*50}")
    print("Training single player model...")

    # Create environment
    render_mode = "ansi" if render_training else None
    env = gym.make('DragonGate-v0', render_mode=render_mode, num_players=1)

    # Create and train model
    model = A2C("MultiInputPolicy", env, verbose=verbose)
    model.learn(total_timesteps=total_timesteps)

    # Evaluate the model
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Single player evaluation: {mean_reward:.2f} ± {std_reward:.2f}")

    # Save the model
    os.makedirs("models", exist_ok=True)
    model.save("models/single_player_model")

    env.close()
    return model

def train_with_simulated_opponents(total_timesteps=5000, render_training=False, verbose=1):
    """Train a model with simulated opponents"""
    print(f"\n{'='*50}")
    print("Training model with simulated opponents...")

    # Create environment with simulated opponents
    render_mode = "ansi" if render_training else None
    env = gym.make('DragonGate-v0', render_mode=render_mode, num_players=4,
                   simulated_players=True)

    # Create and train model
    model = PPO("MultiInputPolicy", env, verbose=verbose)
    model.learn(total_timesteps=total_timesteps)

    # Evaluate the model
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Evaluation with simulated opponents: {mean_reward:.2f} ± {std_reward:.2f}")

    # Save the model
    os.makedirs("models", exist_ok=True)
    model.save("models/multiplayer_model")

    env.close()
    return model

def visualize_model(model, num_players=1, simulated_players=False, num_rounds=5):
    """Run a visualization of the model's behavior"""
    print(f"\n{'='*50}")
    print(f"Visualizing model behavior (players={num_players}, simulated={simulated_players})...")

    # Create environment for visualization
    env = gym.make('DragonGate-v0', render_mode="human", num_players=num_players,
                   simulated_players=simulated_players)

    # Reset environment and visualize multiple rounds
    obs, _ = env.reset()
    total_reward = 0

    for i in range(num_rounds):
        print(f"\nRound {i+1}")
        # Get model's action
        action, _ = model.predict(obs, deterministic=True)

        # Display what action the model is taking
        bet_pct = int(action[0] * 100)
        choice = "Higher" if action[1] > 0.5 else "Lower"
        print(f"Action: Bet {bet_pct}%, Choice: {choice}")

        # Execute action in environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        print(f"Reward: {reward}, Total: {total_reward}")

        # Slow down to make visualization visible
        time.sleep(1)

        if terminated or truncated:
            print("Game Over!")
            break

    env.close()

def compare_models():
    """Compare the performance of different models"""
    print(f"\n{'='*50}")
    print("Loading models for comparison...")

    # Load both models
    try:
        single_model = A2C.load("models/single_player_model")
        multi_model = PPO.load("models/multiplayer_model")
    except:
        print("Models not found. Please train models first.")
        return

    # Create test environments
    env_single = gym.make('DragonGate-v0', render_mode=None, num_players=1)
    env_multi = gym.make('DragonGate-v0', render_mode=None, num_players=4, simulated_players=True)

    # Test single-player model in both environments
    print("\nTesting single-player model...")
    single_in_single, _ = evaluate_policy(single_model, env_single, n_eval_episodes=10)
    single_in_multi, _ = evaluate_policy(single_model, env_multi, n_eval_episodes=10)

    # Test multi-player model in both environments
    print("\nTesting multi-player model...")
    multi_in_single, _ = evaluate_policy(multi_model, env_single, n_eval_episodes=10)
    multi_in_multi, _ = evaluate_policy(multi_model, env_multi, n_eval_episodes=10)

    # Display results
    print("\nModel Performance Comparison:")
    print(f"                      | Single Environment | Multiplayer Environment")
    print(f"----------------------|-------------------|----------------------")
    print(f"Single-player model   | {single_in_single:17.2f} | {single_in_multi:22.2f}")
    print(f"Multi-player model    | {multi_in_single:17.2f} | {multi_in_multi:22.2f}")

    # Create bar chart
    models = ['Single-player', 'Multi-player']
    single_env = [single_in_single, multi_in_single]
    multi_env = [single_in_multi, multi_in_multi]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, single_env, width, label='Single Environment')
    ax.bar(x + width/2, multi_env, width, label='Multiplayer Environment')

    ax.set_ylabel('Average Reward')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()

    plt.savefig('model_comparison.png')
    plt.close()

    print("Comparison chart saved as 'model_comparison.png'")

    env_single.close()
    env_multi.close()

def main():
    print("Dragon Gate Training Demonstration")
    print("==================================")

    # Quick training for demonstration
    training_steps = 8000  # Small for demo purposes
    render_during_training = False
    verbose = 1

    # Train models
    single_model = train_single_player(
        total_timesteps=training_steps,
        render_training=render_during_training,
        verbose=verbose
    )

    multi_model = train_with_simulated_opponents(
        total_timesteps=training_steps,
        render_training=render_during_training,
        verbose=verbose
    )

    # Visualize model behavior
    visualize_model(single_model, num_players=1, simulated_players=False, num_rounds=3)
    visualize_model(multi_model, num_players=4, simulated_players=True, num_rounds=3)

    # Compare model performance
    compare_models()

    print(f"\n{'='*50}")
    print("Demonstration complete!")

if __name__ == "__main__":
    main()