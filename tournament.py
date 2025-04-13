import gymnasium as gym
import numpy as np
import time
import os
from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from dragon_gate_env import DragonGateEnv
import matplotlib.pyplot as plt
from matplotlib.table import Table
import pandas as pd

# Register the custom environment
gym.register(
    id='DragonGate-v0',
    entry_point='dragon_gate_env:DragonGateEnv',
)

class Agent:
    def __init__(self, name, model_cls, policy_type, learning_rate=0.0003, gamma=0.99, n_steps=5, **kwargs):
        """
        Initialize an agent with a specified model and hyperparameters.

        Args:
            name (str): Unique identifier for the agent
            model_cls: SB3 model class (A2C, PPO, etc.)
            policy_type (str): Policy network type (e.g., "MultiInputPolicy")
            learning_rate (float): Learning rate for the model
            gamma (float): Discount factor
            n_steps (int): Number of steps to use in A2C
            **kwargs: Additional hyperparameters for the model
        """
        self.name = name
        self.model_cls = model_cls
        self.policy_type = policy_type
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.n_steps = n_steps
        self.kwargs = kwargs
        self.model = None
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.total_reward = 0
        self.is_trained = False

    def train(self, env, total_timesteps=10000, verbose=0):
        """Train the agent on the given environment"""
        if issubclass(self.model_cls, A2C):
            self.model = self.model_cls(
                self.policy_type,
                env,
                learning_rate=self.learning_rate,
                gamma=self.gamma,
                n_steps=self.n_steps,
                verbose=verbose,
                **self.kwargs
            )
        elif issubclass(self.model_cls, PPO):
            self.model = self.model_cls(
                self.policy_type,
                env,
                learning_rate=self.learning_rate,
                gamma=self.gamma,
                n_steps=self.n_steps,
                verbose=verbose,
                **self.kwargs
            )
        elif issubclass(self.model_cls, DQN):
            self.model = self.model_cls(
                self.policy_type,
                env,
                learning_rate=self.learning_rate,
                gamma=self.gamma,
                verbose=verbose,
                **self.kwargs
            )
        else:
            raise ValueError(f"Unsupported model class: {self.model_cls}")

        print(f"Training agent {self.name}...")
        self.model.learn(total_timesteps=total_timesteps)
        self.is_trained = True
        print(f"Agent {self.name} training completed.")

    def save(self, directory):
        """Save the agent's model"""
        if not self.is_trained:
            raise ValueError("Agent must be trained before saving")
        os.makedirs(directory, exist_ok=True)
        self.model.save(os.path.join(directory, f"{self.name}.zip"))

    def load(self, directory):
        """Load the agent's model"""
        if issubclass(self.model_cls, A2C):
            self.model = A2C.load(os.path.join(directory, f"{self.name}.zip"))
        elif issubclass(self.model_cls, PPO):
            self.model = PPO.load(os.path.join(directory, f"{self.name}.zip"))
        elif issubclass(self.model_cls, DQN):
            self.model = DQN.load(os.path.join(directory, f"{self.name}.zip"))
        else:
            raise ValueError(f"Unsupported model class: {self.model_cls}")
        self.is_trained = True

    def predict(self, observation):
        """Make a prediction based on the observation"""
        if not self.is_trained:
            raise ValueError("Agent must be trained before prediction")
        return self.model.predict(observation)

    def update_stats(self, reward):
        """Update agent statistics based on match results"""
        self.total_reward += reward
        if reward > 0:
            self.wins += 1
        elif reward < 0:
            self.losses += 1
        else:
            self.draws += 1

class Tournament:
    def __init__(self, env_id="DragonGate-v0", render_mode="human"):
        """Initialize a tournament with the specified environment"""
        self.env_id = env_id
        self.render_mode = render_mode
        self.agents = []
        self.match_results = []

    def add_agent(self, agent):
        """Add an agent to the tournament"""
        self.agents.append(agent)

    def train_agents(self, total_timesteps=10000, verbose=0, adversarial_training=False):
        """
        Train all agents in the tournament

        Args:
            total_timesteps: Number of steps to train each agent
            verbose: Verbosity level
            adversarial_training: If True, agents train against other agents;
                                  If False, agents train against the environment
        """
        if not adversarial_training:
            # Standard training against the environment
            for agent in self.agents:
                env = gym.make(self.env_id, render_mode=None)  # No rendering during training
                agent.train(env, total_timesteps=total_timesteps, verbose=verbose)
                env.close()
        else:
            # Adversarial training where agents play against each other
            if len(self.agents) < 2:
                raise ValueError("Need at least 2 agents for adversarial training")

            # First, give each agent some basic training to learn the rules
            for agent in self.agents:
                env = gym.make(self.env_id, render_mode=None)
                agent.train(env, total_timesteps=total_timesteps // 5, verbose=verbose)  # Brief initial training
                env.close()

            # Number of adversarial matches each agent will play
            matches_per_agent = total_timesteps // 100  # Approximate conversion from steps to matches

            print(f"Starting adversarial training with {matches_per_agent} matches per agent pair...")

            # Create a shared environment for agent matches
            env = gym.make(self.env_id, render_mode=None, num_players=2)

            # Each agent plays against all other agents
            for _ in range(3):  # Multiple rounds of training
                for i in range(len(self.agents)):
                    for j in range(len(self.agents)):
                        if i == j:  # Skip self-play for simplicity
                            continue

                        agent1 = self.agents[i]
                        agent2 = self.agents[j]

                        if verbose > 0:
                            print(f"Training {agent1.name} against {agent2.name}...")

                        # Play several matches between these agents
                        for _ in range(matches_per_agent // len(self.agents)):
                            agent1_money = 1000
                            agent2_money = 1000
                            pot = 400

                            # Run a training match
                            for round in range(10):  # Short matches for training
                                if agent1_money <= 0 or agent2_money <= 0 or pot <= 0:
                                    break

                                # Agent 1's turn
                                obs, _ = env.reset()
                                env.money = agent1_money
                                env.pot = pot

                                # Agent 1 makes a decision
                                action1, _ = agent1.predict(obs)
                                obs, reward1, terminated, truncated, info = env.step(action1)

                                # Extract updated state
                                agent1_money = info['money']
                                pot = info['pot']

                                # Collect this experience for agent1's replay buffer
                                # This would require modifying the SB3 Agent class to support this

                                # Agent 2's turn
                                if agent1_money > 0 and pot > 0:
                                    obs, _ = env.reset()
                                    env.money = agent2_money
                                    env.pot = pot

                                    action2, _ = agent2.predict(obs)
                                    obs, reward2, terminated, truncated, info = env.step(action2)

                                    agent2_money = info['money']
                                    pot = info['pot']

                                    # Collect this experience for agent2's replay buffer

                            # After the match, update both agents with collected experience
                            # For now, we rely on the next explicit training phase

                        # After playing against an opponent, do a quick learning update
                        temp_env = gym.make(self.env_id, render_mode=None)
                        agent1.model.set_env(temp_env)
                        agent1.model.learn(total_timesteps=total_timesteps // 10 // len(self.agents))
                        temp_env.close()

            # Final independent training to consolidate learning
            for agent in self.agents:
                env = gym.make(self.env_id, render_mode=None)
                agent.train(env, total_timesteps=total_timesteps // 5, verbose=verbose)
                env.close()

            env.close()

    def run_match(self, agent1, agent2, num_rounds=10, initial_pot=400, verbose=0):
        """Run a match between two agents"""
        # Create environment with the specified pot and render mode
        if verbose > 0:
            render_mode = self.render_mode
        else:
            render_mode = None

        env = gym.make(self.env_id, render_mode=render_mode, num_players=2,
                       min_bet=100, starting_money=1000, max_rounds=100)

        agent1_money = 1000
        agent2_money = 1000
        pot = initial_pot

        for round in range(num_rounds):
            if agent1_money <= 0 or agent2_money <= 0 or pot <= 0:
                break

            # Agent 1's turn
            obs, _ = env.reset()
            env.money = agent1_money
            env.pot = pot

            # Agent 1 makes a decision
            action1, _ = agent1.predict(obs)

            # Process agent 1's action
            obs, reward1, terminated, truncated, info = env.step(action1)
            agent1_money = info['money']
            pot = info['pot']

            if verbose > 0:
                print(f"Round {round+1}/{num_rounds} - {agent1.name} action: {action1}")
                print(f"  Result: {reward1:+d}, Money: {agent1_money}, Pot: {pot}")

            # Agent 2's turn (if game not over)
            if agent1_money > 0 and pot > 0:
                # Reset environment with new values
                obs, _ = env.reset()
                env.money = agent2_money
                env.pot = pot

                # Agent 2 makes a decision
                action2, _ = agent2.predict(obs)

                # Process agent 2's action
                obs, reward2, terminated, truncated, info = env.step(action2)
                agent2_money = info['money']
                pot = info['pot']

                if verbose > 0:
                    print(f"Round {round+1}/{num_rounds} - {agent2.name} action: {action2}")
                    print(f"  Result: {reward2:+d}, Money: {agent2_money}, Pot: {pot}")

        # Calculate overall results
        agent1_final_reward = agent1_money - 1000
        agent2_final_reward = agent2_money - 1000

        # Update agent stats
        agent1.update_stats(agent1_final_reward)
        agent2.update_stats(agent2_final_reward)

        # Record match result
        match_result = {
            'agent1': agent1.name,
            'agent2': agent2.name,
            'agent1_final_money': agent1_money,
            'agent2_final_money': agent2_money,
            'agent1_reward': agent1_final_reward,
            'agent2_reward': agent2_final_reward,
            'rounds_played': round + 1,
            'final_pot': pot
        }

        self.match_results.append(match_result)

        env.close()
        return match_result

    def run_tournament(self, num_rounds=10, verbose=0):
        """Run a tournament where each agent plays against all other agents"""
        if len(self.agents) < 2:
            raise ValueError("Need at least 2 agents for a tournament")

        # Make sure all agents are trained
        for agent in self.agents:
            if not agent.is_trained:
                raise ValueError(f"Agent {agent.name} must be trained before running tournament")

        print("Starting tournament...")
        # Each agent plays against all other agents
        for i in range(len(self.agents)):
            for j in range(i+1, len(self.agents)):
                agent1 = self.agents[i]
                agent2 = self.agents[j]

                print(f"\nMatch: {agent1.name} vs {agent2.name}")
                match_result = self.run_match(agent1, agent2, num_rounds=num_rounds, verbose=verbose)

                # Print match summary
                print(f"Match completed. Results:")
                print(f"  {agent1.name}: ${match_result['agent1_final_money']} (${match_result['agent1_reward']:+d})")
                print(f"  {agent2.name}: ${match_result['agent2_final_money']} (${match_result['agent2_reward']:+d})")
                print(f"  Rounds played: {match_result['rounds_played']}")
                print(f"  Final pot: ${match_result['final_pot']}")

        print("\nTournament completed!")
        self.print_tournament_results()

    def print_tournament_results(self):
        """Print the results of the tournament"""
        print("\n===== Tournament Results =====")
        print(f"Number of agents: {len(self.agents)}")
        print(f"Number of matches: {len(self.match_results)}")

        # Print standings
        agent_stats = []
        for agent in self.agents:
            stats = {
                'name': agent.name,
                'wins': agent.wins,
                'losses': agent.losses,
                'draws': agent.draws,
                'total_reward': agent.total_reward,
                'avg_reward': agent.total_reward / max(1, agent.wins + agent.losses + agent.draws)
            }
            agent_stats.append(stats)

        # Sort by total reward
        agent_stats.sort(key=lambda x: x['total_reward'], reverse=True)

        print("\nStandings:")
        for i, stats in enumerate(agent_stats):
            print(f"{i+1}. {stats['name']}: {stats['wins']} wins, {stats['losses']} losses, " +
                  f"{stats['draws']} draws, Total: ${stats['total_reward']:+d}, " +
                  f"Avg: ${stats['avg_reward']:.2f}")

    def plot_results(self):
        """Visualize tournament results"""
        if not self.match_results:
            print("No match results to plot")
            return

        # Create a dataframe for easier manipulation
        df = pd.DataFrame(self.match_results)

        # Prepare data for the results table
        agent_names = [agent.name for agent in self.agents]
        results_table = np.zeros((len(agent_names), len(agent_names)))

        # Fill in the results matrix
        for match in self.match_results:
            idx1 = agent_names.index(match['agent1'])
            idx2 = agent_names.index(match['agent2'])

            # Store result as win margin
            results_table[idx1, idx2] = match['agent1_reward']
            results_table[idx2, idx1] = match['agent2_reward']

        # Create the heatmap figure
        plt.figure(figsize=(10, 8))

        # Draw the heatmap
        heatmap = plt.imshow(results_table, cmap='RdYlGn')
        plt.colorbar(heatmap, label='Reward')

        # Add text annotations
        for i in range(len(agent_names)):
            for j in range(len(agent_names)):
                if i != j:  # Skip diagonal (no matches against self)
                    plt.text(j, i, f"{results_table[i, j]:+.0f}",
                             ha="center", va="center", color="black")
                else:
                    plt.text(j, i, "X", ha="center", va="center", color="black")

        # Set the tick labels
        plt.xticks(np.arange(len(agent_names)), agent_names, rotation=45)
        plt.yticks(np.arange(len(agent_names)), agent_names)

        plt.title("Tournament Results Heatmap")
        plt.xlabel("Opponent")
        plt.ylabel("Agent")
        plt.tight_layout()

        # Save the figure
        plt.savefig("tournament_results.png")
        plt.close()

        # Create bar chart of total rewards
        agent_rewards = {agent.name: agent.total_reward for agent in self.agents}
        names = list(agent_rewards.keys())
        values = list(agent_rewards.values())

        plt.figure(figsize=(10, 6))
        bars = plt.bar(names, values)

        # Color bars based on sign
        for i, v in enumerate(values):
            if v > 0:
                bars[i].set_color('green')
            else:
                bars[i].set_color('red')

        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.title("Total Rewards by Agent")
        plt.xlabel("Agent")
        plt.ylabel("Total Reward ($)")
        plt.xticks(rotation=45)

        # Add value labels on top of bars
        for i, v in enumerate(values):
            plt.text(i, v + np.sign(v) * 5, f"{v:+d}",
                    ha='center', va='bottom' if v >= 0 else 'top')

        plt.tight_layout()
        plt.savefig("agent_rewards.png")
        plt.close()

        print("Tournament visualization saved as 'tournament_results.png' and 'agent_rewards.png'")

def main():
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

    # Train all agents
    tournament.train_agents(total_timesteps=20000, verbose=1)

    # Run tournament
    tournament.run_tournament(num_rounds=20, verbose=1)

    # Plot and save results
    tournament.plot_results()

if __name__ == "__main__":
    main()
