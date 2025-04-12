# Dragon Gate Agent Training (射龍門)

This project provides a reinforcement learning implementation of Dragon Gate (射龍門), a traditional gambling card game commonly played during Lunar New Year.

## Vibe coding

I basically don't understand what's going on, the purpose of this project is to generate the model for playing dragon gate.

90% of this project is done by `Claude Code`

## Reference

The `dragon_gate_env.py` is basically copied from [here](https://gymnasium.farama.org/environments/toy_text/blackjack/#information)

The font and img both download from [here](https://github.com/openai/gym/tree/master/gym/envs/toy_text)

## Game Rules

Dragon Gate is a card game where players bet on whether a third card will fall between two initial cards:

1. Two cards are dealt to the player
2. Player places a bet (min bet is 100, max bet is the current pot)
3. A third card is dealt
4. If the third card's value is between the two initial cards:
   - Player wins the amount they bet from the pot
5. If the third card's value is outside the two initial cards:
   - Player loses their bet, which is added to the pot
6. If the third card's value is equal to either of the initial cards (hitting the post/撞柱):
   - Player loses twice their bet, which is added to the pot
7. If the two initial cards have the same value:
   - Player must choose whether the third card will be higher or lower
   - If correct, player wins their bet
   - If incorrect, player loses their bet
   - If the third card matches (hitting the post), player loses 3x their bet

## Installation

### Prerequisites

- Python 3.7+
- pip (Python package manager)

### Setup

1. Clone this repository

   ```bash
   git clone <repository-url>
   cd stable-baseline3
   ```

2. Install required dependencies

   ```bash
   pip install numpy gymnasium stable-baselines3 matplotlib pandas pygame
   ```

## Project Components

- `blackjack_env.py`: A reference Gymnasium environment for Blackjack
- `dragon_gate_env.py`: Custom Gymnasium environment for Dragon Gate
- `single_demo.py`: Basic script for training and testing a single agent
- `tournament.py`: Framework for multi-agent tournaments
- `tournament_demo.py`: Demonstration of tournament capabilities

## Usage

### Training a Single Agent

To train and test a single RL agent:

```bash
python single_demo.py
```

This will train an A2C agent on the Dragon Gate environment and visualize its performance.

### Multi-Agent Tournament

For more advanced multi-agent tournaments:

```bash
# Quick demo (fast training, few rounds)
python tournament_demo.py --mode quick

# Full tournament (longer training, more rounds)
python tournament_demo.py --mode full

# Load saved agents for a demo
python tournament_demo.py --mode load

# Visualize agent behavior in specific scenarios
python tournament_demo.py --mode visualize
```

The tournament system allows you to:

1. Train multiple agents with different algorithms (A2C, PPO, DQN)
2. Have agents compete against each other
3. Analyze which strategies perform best
4. Visualize the results through heatmaps and charts

## Visualization

When running a full tournament, the system generates:

- `tournament_results.png`: Heatmap of agent performance against each other
- `agent_rewards.png`: Bar chart showing total rewards earned by each agent

## Extending the Project

You can customize the tournament by:

- Adding new agent types with different hyperparameters
- Creating your own tournament scenarios
- Modifying the Dragon Gate environment parameters

Example of adding a custom agent in `tournament_demo.py`:

```python
Agent("Custom-Strategy", PPO, "MultiInputPolicy",
      learning_rate=0.0002, gamma=0.97,
      clip_range=0.15, ent_coef=0.008)
```

## Troubleshooting

If you encounter rendering issues:

- The system will automatically fall back to text-based rendering if pygame is not installed
- Use `--mode quick` for faster development and testing
- Make sure the card images in the `img/` directory are available

## License

This project is available under the MIT License.
