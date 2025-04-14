import os
import gymnasium as gym
import time
import numpy as np
from tournament import Tournament, Agent
from stable_baselines3 import A2C, PPO, DQN
from dragon_gate_env import DragonGateEnv

# Register the custom environment
gym.register(
    id='DragonGate-v0',
    entry_point='dragon_gate_env:DragonGateEnv',
)

def create_quick_tournament():
    # Create a tournament with fewer timesteps for quicker testing
    tournament = Tournament(env_id="DragonGate-v0", render_mode="ansi")

    # Create simpler agents for quicker training
    agents = [
        Agent("A2C-Default", A2C, "MultiInputPolicy"),
        Agent("A2C-Aggressive", A2C, "MultiInputPolicy", learning_rate=0.001, gamma=0.99),
    ]

    # Add agents to tournament
    for agent in agents:
        tournament.add_agent(agent)

    # Train all agents with minimal timesteps
    tournament.train_agents(total_timesteps=5000, verbose=1)

    # Run a short tournament
    tournament.run_tournament(num_rounds=5, verbose=1)

    return tournament

def create_full_tournament():
    # Create the tournament
    tournament = Tournament(env_id="DragonGate-v0", render_mode="ansi")

    # Create different agents with varied strategies and hyperparameters
    agents = [
        Agent("A2C-Bold", A2C, "MultiInputPolicy", learning_rate=0.001, gamma=0.99, n_steps=5),
        Agent("A2C-Cautious", A2C, "MultiInputPolicy", learning_rate=0.0005, gamma=0.95, n_steps=8),
        Agent("PPO-Aggressive", PPO, "MultiInputPolicy", learning_rate=0.0003, gamma=0.99, n_steps=2048,
              clip_range=0.2, ent_coef=0.01),
        Agent("PPO-Balanced", PPO, "MultiInputPolicy", learning_rate=0.0001, gamma=0.98, n_steps=1024,
              clip_range=0.1, ent_coef=0.005),
    ]

    # Add agents to tournament
    for agent in agents:
        tournament.add_agent(agent)

    # Train all agents with adversarial training
    tournament.train_agents(total_timesteps=20000, verbose=1, adversarial_training=True)

    # Save the trained agents
    for agent in agents:
        agent.save("models")

    # Run tournament
    tournament.run_tournament(num_rounds=20, verbose=1)

    return tournament

def test_saved_agents():
    """Test loading and running a tournament with saved agents"""
    # Create the tournament
    tournament = Tournament(env_id="DragonGate-v0", render_mode="human")

    # Create agents with the same configurations as before
    agents = [
        Agent("A2C-Bold", A2C, "MultiInputPolicy"),
        Agent("A2C-Cautious", A2C, "MultiInputPolicy"),
        Agent("PPO-Aggressive", PPO, "MultiInputPolicy"),
        Agent("PPO-Balanced", PPO, "MultiInputPolicy"),
    ]

    # Load pre-trained models
    for agent in agents:
        try:
            agent.load("models")
            tournament.add_agent(agent)
            print(f"Loaded agent: {agent.name}")
        except:
            print(f"Failed to load agent: {agent.name}")

    # Run a demonstration match with visualization
    if len(tournament.agents) >= 2:
        agent1 = tournament.agents[0]
        agent2 = tournament.agents[1]
        print(f"\nDemo Match: {agent1.name} vs {agent2.name}")
        match_result = tournament.run_match(agent1, agent2, num_rounds=5, verbose=2)

        print(f"\nMatch completed. Results:")
        print(f"  {agent1.name}: ${match_result['agent1_final_money']} (${match_result['agent1_reward']:+d})")
        print(f"  {agent2.name}: ${match_result['agent2_final_money']} (${match_result['agent2_reward']:+d})")
    else:
        print("Not enough agents to run a demo match")

def visualize_agent_behavior():
    """Visualize how different agents behave in various scenarios"""
    # Create the environment
    env = gym.make("DragonGate-v0", render_mode="human", num_players=2)

    # Create different types of agents
    agents = [
        Agent("A2C-Default", A2C, "MultiInputPolicy"),
        Agent("PPO-Default", PPO, "MultiInputPolicy"),
    ]

    # Train agents (short training for demo)
    for agent in agents:
        agent.train(env, total_timesteps=5000, verbose=1)

    # Define test scenarios
    scenarios = [
        {"name": "Close Cards", "card1": 5, "card2": 7},
        {"name": "Wide Range", "card1": 2, "card2": 10},
        {"name": "Same Cards", "card1": 8, "card2": 8},
        {"name": "Extreme Low", "card1": 1, "card2": 3},
        {"name": "Extreme High", "card1": 11, "card2": 13},
    ]

    # Test each agent on each scenario
    for agent in agents:
        print(f"\nTesting {agent.name}")

        for scenario in scenarios:
            print(f"\nScenario: {scenario['name']}")

            # Reset the environment
            obs, _ = env.reset()

            # Override the cards for this scenario
            env.card1 = scenario["card1"]
            env.card2 = scenario["card2"]
            env.card1_suit = np.random.choice(["C", "D", "H", "S"])
            env.card2_suit = np.random.choice(["C", "D", "H", "S"])

            # Get observation based on the scenario cards
            obs = env._get_obs()

            # Get the agent's action
            action, _ = agent.predict(obs)

            # Display the action
            bet_pct = int(action[0] * 100)
            choice = "Higher" if action[1] > 0.5 else "Lower"
            print(f"Cards: {env.card1_suit}{env.card_values[env.card1]} and {env.card2_suit}{env.card_values[env.card2]}")
            print(f"Action: Bet {bet_pct}%, Choice: {choice}")

            # Execute the action to see the result
            obs, reward, terminated, truncated, info = env.step(action)

            # Display the result
            print(f"Result: {reward:+d}, Money: {info['money']}, Pot: {info['pot']}")

            # Render the final state
            env.render()

            # Pause to let the user see the result
            time.sleep(2)

    env.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dragon Gate Tournament Demo")
    parser.add_argument("--mode", type=str, choices=["quick", "full", "load", "visualize"],
                        default="quick", help="Demo mode to run")

    args = parser.parse_args()

    if args.mode == "quick":
        print("Running quick tournament demo...")
        tournament = create_quick_tournament()
    elif args.mode == "full":
        print("Running full tournament (this may take a while)...")
        tournament = create_full_tournament()
        # Plot and save results
        tournament.plot_results()
    elif args.mode == "load":
        print("Loading saved agents for a demo match...")
        test_saved_agents()
    elif args.mode == "visualize":
        print("Visualizing agent behavior in various scenarios...")
        visualize_agent_behavior()
