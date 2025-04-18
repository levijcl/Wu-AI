from typing import Union
import os
import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel
# Use relative import
from tournament import Agent
from stable_baselines3 import PPO, A2C

app = FastAPI()

# Try to load the agent from models directory
models_dir = "models"
model_names = ["PPO-Aggressive", "A2C-Bold"]
agent = None

# Try to load each model until one is found
for model_name in model_names:
    try:
        if model_name.startswith("PPO"):
            agent = Agent(model_name, PPO, "MultiInputPolicy")
        elif model_name.startswith("A2C"):
            agent = Agent(model_name, A2C, "MultiInputPolicy")

        agent.load(models_dir)
        print(f"Loaded agent: {model_name}")
        break
    except Exception as e:
        print(f"Could not load {model_name}: {e}")
        agent = None

# If no model found, create a temporary model with basic functionality
if agent is None:
    print("No models found, creating a basic agent for demo")
    agent = Agent("Default-Agent", PPO, "MultiInputPolicy")

    # Create a minimal model that will return reasonable actions
    class DummyModel:
        def predict(self, observation):
            # Extract card values
            card1 = observation["card1"][0]
            card2 = observation["card2"][0]
            card_diff = card2 - card1

            # Simple heuristic strategy
            if card_diff <= 3:
                bet_proportion = 0.2  # Low bet for narrow range
            elif card_diff >= 6:
                bet_proportion = 0.6  # Higher bet for wide range
            else:
                bet_proportion = 0.4  # Medium bet for average range

            # For high/low choice (equal cards)
            if card1 == card2:
                if card1 < 7:  # If below middle value
                    high_low_choice = 0.8  # Prefer higher
                else:
                    high_low_choice = 0.2  # Prefer lower
            else:
                high_low_choice = 0.5  # Random for non-equal cards

            return np.array([bet_proportion, high_low_choice]), None

    agent.model = DummyModel()
    agent.is_trained = True

class Env(BaseModel):
    card1: float
    card2: float
    playerMoney: float
    pot: float
    allPlayerMoney: list[float] = None  # Optional field for all player money


@app.post("/predict")
def update_item(env: Env):
    # Convert the environment state into the format expected by the model
    obs = {
        "card1": np.array([env.card1], dtype=np.float32),
        "card2": np.array([env.card2], dtype=np.float32),
        "player_money": np.array([env.playerMoney], dtype=np.float32),
        "pot": np.array([env.pot], dtype=np.float32)
    }

    # Add all player money if provided
    if env.allPlayerMoney:
        obs["all_player_money"] = np.array(env.allPlayerMoney, dtype=np.float32)

    # Get prediction from the agent
    action, _ = agent.predict(obs)

    # Return the prediction
    return {
        "bet_percentage": int(action[0] * 100),
        "higher": True if action[1] > 0.5 else False,
        "agent_name": agent.name
    }
