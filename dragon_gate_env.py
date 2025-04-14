import os
from typing import Optional, Dict, Any, Tuple, List
import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled


class DragonGateGame:
    """
    Dragon Gate (射龍門) Game Core Logic

    This class handles the core game mechanics without the Gym environment structure.
    It can be used as the foundation for different types of environments.
    """

    def __init__(self, num_players: int = 4, min_bet: int = 100,
                 starting_money: int = 1000, num_decks: int = 2):
        self.num_players = num_players
        self.min_bet = min_bet
        self.starting_money = starting_money
        self.pot = self.min_bet * self.num_players
        self.player_moneys = [self.starting_money] * self.num_players
        self.current_player_idx = 0
        self.round = 0
        self.num_decks = num_decks

        # Card deck related attributes
        self.deck = []
        self.discard_pile = []

        # Card values and suits
        self.card_values = {
            1: 'A', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7',
            8: '8', 9: '9', 10: 'T', 11: 'J', 12: 'Q', 13: 'K'
        }
        self.card_suits = ["C", "D", "H", "S"]  # Clubs, Diamonds, Hearts, Spades

        # Initialize the game state
        self.initialize_deck()
        self.deal_initial_cards()

    def initialize_deck(self):
        """Initialize a deck with the specified number of standard decks"""
        self.deck = []
        self.discard_pile = []

        # Create multiple decks
        for _ in range(self.num_decks):
            for suit in self.card_suits:
                for value in range(1, 14):  # 1-13 (Ace to King)
                    self.deck.append((value, suit))

        # Shuffle the deck
        np.random.shuffle(self.deck)

    def shuffle_deck(self):
        """Shuffle the current deck"""
        np.random.shuffle(self.deck)

    def draw_card(self):
        """Draw a card from the deck. If deck is empty, shuffle discard pile back in."""
        if not self.deck:
            if not self.discard_pile:
                # This should never happen in normal gameplay, but just in case
                self.initialize_deck()
            else:
                # Move discard pile back to deck and shuffle
                self.deck = self.discard_pile.copy()
                self.discard_pile = []
                self.shuffle_deck()

        # Draw the top card
        card = self.deck.pop(0)
        return card

    def deal_initial_cards(self):
        """Deal two initial cards to the current player"""
        # Deal initial two cards
        card1 = self.draw_card()
        card2 = self.draw_card()

        self.card1, self.card1_suit = card1
        self.card2, self.card2_suit = card2

        # Sort cards so card1 is always the smaller one (simplifies the game logic)
        if self.card1 > self.card2:
            self.card1, self.card2 = self.card2, self.card1
            self.card1_suit, self.card2_suit = self.card2_suit, self.card1_suit

    def reset_game(self, seed=None):
        """Reset the game to initial state"""
        if seed is not None:
            np.random.seed(seed)

        self.pot = self.min_bet * self.num_players
        self.player_moneys = [self.starting_money] * self.num_players
        self.current_player_idx = 0
        self.round = 0

        # Initialize deck
        self.initialize_deck()
        self.deal_initial_cards()

    def process_bet(self, bet_proportion, high_low_choice):
        """
        Process a player's bet and determine the outcome

        Args:
            bet_proportion: Value between 0 and 1 representing bet size
            high_low_choice: 0 for lower, 1 for higher (when cards are equal)

        Returns:
            Tuple of (reward, terminated)
        """
        # Get current player's money
        current_money = self.player_moneys[self.current_player_idx]

        # Check if cards are equal to apply the right penalty multiple
        penalty_multiple = 3 if self.card1 == self.card2 else 2

        # Limit max bet to ensure player won't go into debt even with penalties
        max_bet = min(current_money // penalty_multiple, self.pot)
        max_bet = max(max_bet, 1)  # Ensure max_bet is at least 1

        min_bet_actual = min(self.min_bet, max_bet)  # Ensure min_bet doesn't exceed what player has
        min_bet_actual = max(min_bet_actual, 1)  # Ensure minimum bet is at least 1

        # Scale the normalized bet to the actual range
        if max_bet > min_bet_actual:
            bet_amount = int(min_bet_actual + bet_proportion * (max_bet - min_bet_actual))
        else:
            bet_amount = min_bet_actual

        # Final check to ensure bet is at least 1
        bet_amount = max(bet_amount, 1)

        # Override high/low choice in logical edge cases
        if self.card1 == self.card2:
            if self.card1 == 1:  # If both cards are 1, force "higher" since nothing can be lower
                high_low_choice = 1
            elif self.card1 == 13:  # If both cards are 13 (King), force "lower" since nothing can be higher
                high_low_choice = 0

        # Deal the third card
        card3, card3_suit = self.draw_card()

        # Determine outcome
        reward = 0

        # Check if cards are equal
        if self.card1 == self.card2:
            if (high_low_choice == 1 and card3 > self.card1) or (high_low_choice == 0 and card3 < self.card1):
                # Correct guess
                reward = bet_amount
                current_money += bet_amount
                self.pot -= bet_amount
            elif card3 == self.card1:
                # Hit the post (equal cards case)
                reward = -3 * bet_amount
                current_money -= 3 * bet_amount
                self.pot += 3 * bet_amount
            else:
                # Wrong guess
                reward = -bet_amount
                current_money -= bet_amount
                self.pot += bet_amount
        else:
            # Normal case: cards are different
            if self.card1 < card3 < self.card2:
                # Card is between the two cards
                reward = bet_amount
                current_money += bet_amount
                self.pot -= bet_amount
            elif card3 == self.card1 or card3 == self.card2:
                # Hit the post
                reward = -2 * bet_amount
                current_money -= 2 * bet_amount
                self.pot += 2 * bet_amount
            else:
                # Card is outside the range
                reward = -bet_amount
                current_money -= bet_amount
                self.pot += bet_amount

        # Update player's money
        self.player_moneys[self.current_player_idx] = current_money

        # Save the third card and other info
        self.card3 = card3
        self.card3_suit = card3_suit
        self.last_bet = bet_amount
        self.last_reward = reward
        self.high_low_choice = high_low_choice

        # Add used cards to discard pile
        self.discard_pile.append((self.card1, self.card1_suit))
        self.discard_pile.append((self.card2, self.card2_suit))
        self.discard_pile.append((card3, card3_suit))

        # Check if game is over for this player
        terminated = current_money <= 0 or self.pot <= 0

        # If not terminated, prepare for next round
        if not terminated:
            # Move to next player
            self.next_player()
            # Deal new cards
            self.deal_initial_cards()
            # Increment round counter if we've gone through all players
            if self.current_player_idx == 0:
                self.round += 1

        return reward, terminated

    def next_player(self):
        """Move to the next player in the rotation"""
        self.current_player_idx = (self.current_player_idx + 1) % self.num_players


class DragonGateEnv(gym.Env):
    """
    Dragon Gate (射龍門) Environment

    This is a card game where the player is dealt two cards, and then bets on whether
    a third card will fall between the two initial cards.

    ### Rules:
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

    ### Action Space
    The action is a tuple with two values:
    - bet_amount: The amount to bet (normalized between 0 and 1, will be scaled to min-max)
    - high_low_choice: Only used when the two initial cards are the same
      - 0: Bet that the third card will be lower
      - 1: Bet that the third card will be higher

    ### Observation Space
    The observation is a dictionary with:
    - card1: First card value (1-13)
    - card2: Second card value (1-13)
    - pot: Current pot amount (normalized)
    - player_money: Player's current money (normalized)

    ### Rewards
    - Winning: +bet_amount
    - Losing: -bet_amount
    - Hitting the post: -2*bet_amount or -3*bet_amount if the two initial cards are the same

    ### Starting State
    Each player starts with 1000 units of money.
    The initial pot is 100 * number_of_players.

    ### Episode End
    The episode ends when the player runs out of money or reaches a maximum number of rounds.
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "ansi"],
        "render_fps": 4,
    }

    def __init__(self, render_mode: Optional[str] = None, num_players: int = 4, min_bet: int = 100,
                 starting_money: int = 1000, max_rounds: int = 100, num_decks: int = 2,
                 simulated_players: bool = False):
        self.max_rounds = max_rounds
        self.simulated_players = simulated_players

        # Create the actual game instance
        self.game = DragonGateGame(
            num_players=num_players,
            min_bet=min_bet,
            starting_money=starting_money,
            num_decks=num_decks
        )

        # For backward compatibility
        self.num_players = self.game.num_players
        self.min_bet = self.game.min_bet
        self.starting_money = self.game.starting_money
        self.player_moneys = self.game.player_moneys
        self.current_player_idx = self.game.current_player_idx
        self.money = self.game.player_moneys[self.current_player_idx]
        self.round = self.game.round
        self.pot = self.game.pot

        # Define action space: (bet_amount, high_low_choice)
        # bet_amount is normalized between 0-1, will be scaled to actual min-max
        # high_low_choice: 0 = bet lower, 1 = bet higher (only used when cards are equal)
        self.action_space = spaces.Box(
            low=np.array([0, 0]),
            high=np.array([1, 1]),
            dtype=np.float32
        )

        # Define observation space
        obs_dict = {
            # Cards are 1-13 (Ace=1, Jack=11, Queen=12, King=13)
            "card1": spaces.Box(low=1, high=13, shape=(1,), dtype=np.float32),
            "card2": spaces.Box(low=1, high=13, shape=(1,), dtype=np.float32),
            "pot": spaces.Box(low=0, high=float('inf'), shape=(1,), dtype=np.float32),
            "player_money": spaces.Box(low=0, high=float('inf'), shape=(1,), dtype=np.float32)
        }

        self.observation_space = spaces.Dict(obs_dict)

        self.render_mode = render_mode

        # Initialize pygame and load card images
        self.pygame = None
        self.card_images = {}
        self.hidden_card_img = None

        if self.render_mode in ["human", "rgb_array"]:
            try:
                self._init_pygame()
                self._load_card_images()
            except DependencyNotInstalled:
                print("Warning: pygame not available, falling back to text rendering")
                self.render_mode = "ansi"
            except Exception as e:
                print(f"Error initializing rendering: {e}")
                self.render_mode = "ansi"

    def _get_obs(self):
        # Update local copies from the game object
        self.player_moneys = self.game.player_moneys
        self.current_player_idx = self.game.current_player_idx
        self.money = self.game.player_moneys[self.current_player_idx]
        self.pot = self.game.pot
        self.round = self.game.round

        # Access the card values from the game
        self.card1 = self.game.card1
        self.card2 = self.game.card2
        self.card1_suit = self.game.card1_suit
        self.card2_suit = self.game.card2_suit

        # Create observation dict
        obs = {
            "card1": np.array([self.card1], dtype=np.float32),
            "card2": np.array([self.card2], dtype=np.float32),
            "pot": np.array([self.pot], dtype=np.float32),
            "player_money": np.array([self.money], dtype=np.float32)
        }

        return obs

    def _get_info(self):
        return {
            "round": self.round,
            "pot": self.pot,
            "money": self.money,
            "current_player_idx": self.current_player_idx,
            "all_player_money": self.player_moneys
        }

    def _simulate_other_players(self):
        """
        If simulated_players is enabled, execute actions for other players
        with a simple rule-based strategy.
        """
        if not self.simulated_players:
            return

        # Save the current player index
        original_player = self.game.current_player_idx

        # Simple rule-based action for other players
        for _ in range(self.num_players - 1):
            # Skip if we're back to the original player
            if self.game.current_player_idx == original_player:
                break

            # Skip if this player is out of money
            if self.game.player_moneys[self.game.current_player_idx] <= 0:
                self.game.next_player()
                continue

            # Simple rule-based strategy:
            # 1. If cards are close together (difference <= 3), bet minimally
            # 2. If wide range (difference > 6), bet 50% of max
            # 3. Otherwise, bet 30% of max
            card_diff = self.game.card2 - self.game.card1

            if card_diff <= 3:
                bet_proportion = 0.1  # Minimum bet
            elif card_diff > 6:
                bet_proportion = 0.5  # Half max bet
            else:
                bet_proportion = 0.3  # Medium bet

            # For high/low choice, use probability-based approach
            if self.game.card1 == self.game.card2:
                # If cards are equal, choose direction with more cards
                middle_value = 7  # Middle card value
                if self.game.card1 < middle_value:
                    high_low_choice = 1  # Higher has more cards
                else:
                    high_low_choice = 0  # Lower has more cards
            else:
                high_low_choice = 0.5  # Not used

            # Execute the action
            self.game.process_bet(bet_proportion, high_low_choice)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # Reset the internal game state
        if seed is not None:
            self.np_random = np.random.RandomState(seed)

        self.game.reset_game(seed)

        # Sync our variables with the game
        self.pot = self.game.pot
        self.player_moneys = self.game.player_moneys
        self.current_player_idx = self.game.current_player_idx
        self.money = self.game.player_moneys[self.current_player_idx]
        self.round = self.game.round

        # Copy card information for rendering
        self.card1 = self.game.card1
        self.card2 = self.game.card2
        self.card1_suit = self.game.card1_suit
        self.card2_suit = self.game.card2_suit

        # Render initial state
        if self.render_mode in ["human", "rgb_array"] and self.pygame is not None:
            self._render_frame()
        elif self.render_mode == "ansi":
            self._render_text()

        return self._get_obs(), self._get_info()

    def step(self, action):
        # Extract action components
        bet_proportion = float(action[0])
        bet_proportion = np.clip(bet_proportion, 0, 1)
        high_low_choice = 1 if action[1] > 0.5 else 0  # 0 = lower, 1 = higher

        # Process the action through the game core
        reward, terminated = self.game.process_bet(bet_proportion, high_low_choice)

        # If we have simulated players and game isn't over, let them play
        if not terminated and self.simulated_players:
            self._simulate_other_players()

        # Update our copies of the game state
        self.card1 = self.game.card1
        self.card2 = self.game.card2
        self.card1_suit = self.game.card1_suit
        self.card2_suit = self.game.card2_suit
        self.pot = self.game.pot
        self.player_moneys = self.game.player_moneys
        self.current_player_idx = self.game.current_player_idx
        self.money = self.game.player_moneys[self.current_player_idx]
        self.round = self.game.round

        # Copy other attributes for rendering
        if hasattr(self.game, 'card3'):
            self.card3 = self.game.card3
            self.card3_suit = self.game.card3_suit
        if hasattr(self.game, 'last_bet'):
            self.last_bet = self.game.last_bet
        if hasattr(self.game, 'last_reward'):
            self.last_reward = self.game.last_reward
        if hasattr(self.game, 'high_low_choice'):
            self.high_low_choice = self.game.high_low_choice

        # Check for additional termination conditions (max rounds)
        if self.round >= self.max_rounds:
            terminated = True

        # Truncated is always False in this environment
        truncated = False

        # Render if needed
        if self.render_mode in ["human", "rgb_array"] and self.pygame is not None:
            self._render_frame()
        elif self.render_mode == "ansi":
            self._render_text()

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _init_pygame(self):
        try:
            import pygame
            self.pygame = pygame
            self.pygame.init()
            self.pygame.font.init()

            if self.render_mode == "human":
                self.screen_width, self.screen_height = 800, 600
                self.screen = self.pygame.display.set_mode((self.screen_width, self.screen_height))
                self.pygame.display.set_caption("Dragon Gate")
                self.clock = self.pygame.time.Clock()
            else:
                self.screen_width, self.screen_height = 800, 600
                self.screen = self.pygame.Surface((self.screen_width, self.screen_height))

            # Load fonts
            try:
                self.font_large = self.pygame.font.Font(os.path.join("font", "Minecraft.ttf"), 28)
                self.font_medium = self.pygame.font.Font(os.path.join("font", "Minecraft.ttf"), 22)
                self.font_small = self.pygame.font.Font(os.path.join("font", "Minecraft.ttf"), 16)
            except:
                self.font_large = self.pygame.font.SysFont('Arial', 28)
                self.font_medium = self.pygame.font.SysFont('Arial', 22)
                self.font_small = self.pygame.font.SysFont('Arial', 16)

        except ImportError:
            self.pygame = None
            raise DependencyNotInstalled("pygame is not installed, run `pip install pygame`")

    def _load_card_images(self):
        """Load card images from the img directory"""
        if self.pygame is None:
            return

        try:
            # The hidden card image
            self.hidden_card_img = self.pygame.image.load(os.path.join("img", "Card.png"))

            # Load all card images
            for suit in self.card_suits:
                for value in range(1, 14):
                    card_key = (suit, value)
                    card_value_str = self.card_values[value]

                    # Construct filename (e.g., "img/CA.png" for Ace of Clubs)
                    file_name = f"{suit}{card_value_str}.png"
                    file_path = os.path.join("img", file_name)

                    if os.path.exists(file_path):
                        self.card_images[card_key] = self.pygame.image.load(file_path)
                    else:
                        print(f"Warning: Card image not found: {file_path}")
        except Exception as e:
            print(f"Error loading card images: {e}")

    def _render_text(self):
        """Render the game state as text (ANSI)"""
        if not hasattr(self, 'card1') or not hasattr(self, 'card2'):
            return ""

        # Get card names with suit
        card1_str = f"{self.card1_suit}{self.game.card_values[self.card1]}"
        card2_str = f"{self.card2_suit}{self.game.card_values[self.card2]}"

        output = "\n" + "="*50 + "\n"
        output += f"DRAGON GATE - Round: {self.round}\n"
        output += f"Pot: {self.pot} | Current Player: {self.current_player_idx + 1} (${self.money})\n"

        # Show all players' money if in multiplayer mode
        if self.num_players > 1:
            output += "All Players Money:\n"
            for i, money in enumerate(self.player_moneys):
                is_current = " (current)" if i == self.current_player_idx else ""
                output += f"  Player {i+1}: ${money}{is_current}\n"

        output += f"Cards in deck: {len(self.game.deck)} | Cards in discard: {len(self.game.discard_pile)}\n"

        # Display the two initial cards
        output += f"\nInitial Cards: {card1_str} and {card2_str}\n"

        # Display the third card and result if available
        if hasattr(self, 'card3') and self.round != 0:
            card3_str = f"{self.card3_suit}{self.game.card_values[self.card3]}"
            output += f"Third Card: {card3_str}\n"

            if hasattr(self, 'last_bet'):
                output += f"Bet: {self.last_bet}\n"

            if hasattr(self, 'last_reward'):
                result = "WON" if self.last_reward > 0 else "LOST"
                output += f"Result: {result} {abs(self.last_reward)}\n"

            # Explain outcome
            if self.card1 == self.card2:
                choice = "higher" if self.high_low_choice == 1 else "lower"
                if (self.high_low_choice == 1 and self.card3 > self.card1) or (self.high_low_choice == 0 and self.card3 < self.card1):
                    output += f"Card was {choice} as predicted!\n"
                elif self.card3 == self.card1:
                    output += f"Hit the post! Triple penalty.\n"
                else:
                    output += f"Card was not {choice} as predicted.\n"
            else:
                if self.card1 < self.card3 < self.card2:
                    output += f"Card was between {self.card1} and {self.card2}!\n"
                elif self.card3 == self.card1 or self.card3 == self.card2:
                    output += f"Hit the post! Double penalty.\n"
                else:
                    output += f"Card was outside the range.\n"

        output += "="*50 + "\n"

        print(output)
        return output

    def _render_frame(self):
        if self.pygame is None:
            return

        # Colors
        bg_color = (7, 99, 36)  # Dark green background like a card table
        white = (255, 255, 255)
        black = (0, 0, 0)
        gold = (212, 175, 55)  # Gold color for pot
        red = (255, 0, 0)
        green = (0, 200, 0)

        # Fill background
        self.screen.fill(bg_color)

        # Draw title
        title = self.font_large.render("Dragon Gate", True, gold)
        self.screen.blit(title, (self.screen_width//2 - title.get_width()//2, 20))

        # Draw cards
        card_width, card_height = 120, 170
        card_spacing = 40
        card_start_x = (self.screen_width - (3 * card_width + 2 * card_spacing)) // 2
        card_y = 80

        # Draw all cards
        cards = [
            (self.card1, self.card1_suit),
            (self.card2, self.card2_suit)
        ]

        # Add third card if available
        if hasattr(self, 'card3'):
            cards.append((self.card3, self.card3_suit))

        for i, (card_value, card_suit) in enumerate(cards):
            x = card_start_x + i * (card_width + card_spacing)
            y = card_y

            # Get card image
            card_key = (card_suit, card_value)
            if card_key in self.card_images:
                # Use the loaded card image
                card_img = self.card_images[card_key]
                card_img = self.pygame.transform.scale(card_img, (card_width, card_height))
                self.screen.blit(card_img, (x, y))
            else:
                # Fallback: draw a placeholder rectangle with text
                self.pygame.draw.rect(self.screen, white, (x, y, card_width, card_height))
                self.pygame.draw.rect(self.screen, black, (x, y, card_width, card_height), 2)

                value_str = self.game.card_values[card_value]
                card_text = self.font_medium.render(f"{card_suit}{value_str}", True, black)
                self.screen.blit(card_text, (x + card_width//2 - card_text.get_width()//2,
                                         y + card_height//2 - card_text.get_height()//2))

        # Draw game info
        info_y = card_y + card_height + 50

        # Draw pot and current player money
        pot_text = self.font_medium.render(f"Pot: ${self.pot}", True, gold)
        self.screen.blit(pot_text, (50, info_y))

        player_text = self.font_medium.render(f"Player {self.current_player_idx + 1}: ${self.money}", True, white)
        self.screen.blit(player_text, (50, info_y + 40))

        # Draw round info
        round_text = self.font_medium.render(f"Round: {self.round}", True, white)
        self.screen.blit(round_text, (50, info_y + 80))

        # Draw all player money if in multiplayer mode
        if self.num_players > 1 and self.num_players <= 8:  # Only show if we have a reasonable number of players
            y_offset = info_y + 120
            players_text = self.font_medium.render("All Players:", True, white)
            self.screen.blit(players_text, (50, y_offset))

            for i, money in enumerate(self.player_moneys):
                color = gold if i == self.current_player_idx else white
                player_money_text = self.font_small.render(f"P{i+1}: ${money}", True, color)
                # Display in two columns if more than 4 players
                if i < 4 or self.num_players <= 4:
                    self.screen.blit(player_money_text, (50, y_offset + 30 + (i % 4) * 25))
                else:
                    self.screen.blit(player_money_text, (150, y_offset + 30 + (i % 4) * 25))

        # Draw card range text
        if self.card1 == self.card2:
            range_text = self.font_medium.render("Cards equal: Bet if next card will be higher or lower", True, white)
        else:
            range_text = self.font_medium.render(f"Target: Card between {self.game.card_values[self.card1]} and {self.game.card_values[self.card2]}", True, white)

        self.screen.blit(range_text, (self.screen_width//2 - range_text.get_width()//2, info_y))

        # Draw bet info if available
        if hasattr(self, 'last_bet'):
            bet_text = self.font_medium.render(f"Bet: ${self.last_bet}", True, white)
            self.screen.blit(bet_text, (self.screen_width - 230, info_y))

            # Show high/low choice if cards were equal
            if self.card1 == self.card2:
                choice = "Higher" if self.high_low_choice > 0.5 else "Lower"
                choice_text = self.font_medium.render(f"Choice: {choice}", True, white)
                self.screen.blit(choice_text, (self.screen_width - 230, info_y + 40))

        # Draw result if available
        if hasattr(self, 'last_reward'):
            result_color = green if self.last_reward > 0 else red
            result_text = self.font_medium.render(f"Result: {self.last_reward:+d}", True, result_color)
            self.screen.blit(result_text, (self.screen_width - 230, info_y + 80))

            # Explain outcome
            outcome_y = info_y + 120
            if self.card1 == self.card2:
                choice = "higher" if self.high_low_choice == 1 else "lower"
                if (self.high_low_choice == 1 and self.card3 > self.card1) or (self.high_low_choice == 0 and self.card3 < self.card1):
                    outcome = f"Card was {choice} as predicted!"
                elif self.card3 == self.card1:
                    outcome = f"Hit the post! Triple penalty."
                else:
                    outcome = f"Card was not {choice} as predicted."
            else:
                if self.card1 < self.card3 < self.card2:
                    outcome = f"Card was within the target range!"
                elif self.card3 == self.card1 or self.card3 == self.card2:
                    outcome = f"Hit the post! Double penalty."
                else:
                    outcome = f"Card was outside the target range."

            outcome_text = self.font_small.render(outcome, True, white)
            self.screen.blit(outcome_text, (self.screen_width//2 - outcome_text.get_width()//2, outcome_y))

        # Display to screen
        if self.render_mode == "human":
            self.pygame.event.pump()
            self.pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

        return np.transpose(
            np.array(self.pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
        ) if self.pygame is not None else None

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.__class__.__name__}", render_mode="rgb_array")'
            )
            return

        if self.render_mode in ["human", "rgb_array"] and self.pygame is not None:
            return self._render_frame()
        elif self.render_mode == "ansi":
            return self._render_text()

    def close(self):
        if hasattr(self, 'pygame') and self.pygame is not None:
            self.pygame.quit()
