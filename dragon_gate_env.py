import os
from typing import Optional, Dict, Any, Tuple, List
import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled


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
                 starting_money: int = 1000, max_rounds: int = 100, num_decks: int = 2):
        self.num_players = num_players
        self.min_bet = min_bet
        self.starting_money = starting_money
        self.max_rounds = max_rounds
        self.pot = self.min_bet * self.num_players * 10
        self.money = self.starting_money
        self.round = 0
        self.num_decks = num_decks

        # Card deck related attributes
        self.deck = []
        self.discard_pile = []

        # Define action space: (bet_amount, high_low_choice)
        # bet_amount is normalized between 0-1, will be scaled to actual min-max
        # high_low_choice: 0 = bet lower, 1 = bet higher (only used when cards are equal)
        self.action_space = spaces.Box(
            low=np.array([0, 0]),
            high=np.array([1, 1]),
            dtype=np.float32
        )

        # Define observation space
        self.observation_space = spaces.Dict({
            # Cards are 1-13 (Ace=1, Jack=11, Queen=12, King=13)
            "card1": spaces.Box(low=1, high=13, shape=(1,), dtype=np.float32),
            "card2": spaces.Box(low=1, high=13, shape=(1,), dtype=np.float32),
            "pot": spaces.Box(low=0, high=float('inf'), shape=(1,), dtype=np.float32),
            "player_money": spaces.Box(low=0, high=float('inf'), shape=(1,), dtype=np.float32)
        })

        self.render_mode = render_mode

        # Card values and suits
        self.card_values = {
            1: 'A', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7',
            8: '8', 9: '9', 10: 'T', 11: 'J', 12: 'Q', 13: 'K'
        }

        self.card_suits = ["C", "D", "H", "S"]  # Clubs, Diamonds, Hearts, Spades

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
        return {
            "card1": np.array([self.card1], dtype=np.float32),
            "card2": np.array([self.card2], dtype=np.float32),
            "pot": np.array([self.pot], dtype=np.float32),
            "player_money": np.array([self.money], dtype=np.float32)
        }

    def _get_info(self):
        return {
            "round": self.round,
            "pot": self.pot,
            "money": self.money
        }

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
        self.shuffle_deck()

    def shuffle_deck(self):
        """Shuffle the current deck"""
        self.np_random.shuffle(self.deck)

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

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.pot = self.min_bet * self.num_players * 10
        self.money = self.starting_money
        self.round = 0

        # Initialize deck if it's the first time
        if not hasattr(self, 'deck') or not self.deck:
            self.initialize_deck()

        # Deal initial two cards
        card1 = self.draw_card()
        card2 = self.draw_card()

        self.card1, self.card1_suit = card1
        self.card2, self.card2_suit = card2

        # Sort cards so card1 is always the smaller one (simplifies the game logic)
        if self.card1 > self.card2:
            self.card1, self.card2 = self.card2, self.card1
            self.card1_suit, self.card2_suit = self.card2_suit, self.card1_suit

        if self.render_mode in ["human", "rgb_array"] and self.pygame is not None:
            self._render_frame()
        elif self.render_mode == "ansi":
            self._render_text()

        return self._get_obs(), self._get_info()

    def step(self, action):
        # Extract and scale bet amount
        bet_proportion = float(action[0])
        bet_proportion = np.clip(bet_proportion, 0, 1)

        max_bet = min(self.money, self.pot)
        min_bet_actual = min(self.min_bet, max_bet)  # Ensure min_bet doesn't exceed what player has

        # Scale the normalized bet to the actual range
        if max_bet > min_bet_actual:
            bet_amount = int(min_bet_actual + bet_proportion * (max_bet - min_bet_actual))
        else:
            bet_amount = min_bet_actual

        # Extract high/low choice for equal cards
        high_low_choice = 1 if action[1] > 0.5 else 0  # 0 = lower, 1 = higher

        # Override high/low choice in logical edge cases
        if self.card1 == self.card2:
            if self.card1 == 1:  # If both cards are 1, force "higher" since nothing can be lower
                high_low_choice = 1
            elif self.card1 == 13:  # If both cards are 13 (King), force "lower" since nothing can be higher
                high_low_choice = 0

        # Deal the third card from the deck
        card3, card3_suit = self.draw_card()

        # Determine outcome
        reward = 0

        # Check if cards are equal
        if self.card1 == self.card2:
            if (high_low_choice == 1 and card3 > self.card1) or (high_low_choice == 0 and card3 < self.card1):
                # Correct guess
                reward = bet_amount
                self.money += bet_amount
                self.pot -= bet_amount
            elif card3 == self.card1:
                # Hit the post (equal cards case)
                reward = -3 * bet_amount
                self.money -= 3 * bet_amount
                self.pot += 3 * bet_amount
            else:
                # Wrong guess
                reward = -bet_amount
                self.money -= bet_amount
                self.pot += bet_amount
        else:
            # Normal case: cards are different
            if self.card1 < card3 < self.card2:
                # Card is between the two cards
                reward = bet_amount
                self.money += bet_amount
                self.pot -= bet_amount
            elif card3 == self.card1 or card3 == self.card2:
                # Hit the post
                reward = -2 * bet_amount
                self.money -= 2 * bet_amount
                self.pot += 2 * bet_amount
            else:
                # Card is outside the range
                reward = -bet_amount
                self.money -= bet_amount
                self.pot += bet_amount

        # Increment round counter
        self.round += 1

        # Check if game is over
        terminated = False
        if self.money <= 0 or self.pot <= 0 or self.round >= self.max_rounds:
            terminated = True

        # Save the third card and other info for rendering
        self.card3 = card3
        self.card3_suit = card3_suit
        self.last_bet = bet_amount
        self.last_reward = reward
        self.high_low_choice = high_low_choice

        if self.render_mode in ["human", "rgb_array"] and self.pygame is not None:
            self._render_frame()
        elif self.render_mode == "ansi":
            self._render_text()

        # Add used cards to discard pile
        self.discard_pile.append((self.card1, self.card1_suit))
        self.discard_pile.append((self.card2, self.card2_suit))
        self.discard_pile.append((card3, card3_suit))

        # Deal new cards for next round if not terminated
        if not terminated:
            # Deal two new cards from the deck
            card1 = self.draw_card()
            card2 = self.draw_card()

            self.card1, self.card1_suit = card1
            self.card2, self.card2_suit = card2

            # Sort cards so card1 is always the smaller one (simplifies the game logic)
            if self.card1 > self.card2:
                self.card1, self.card2 = self.card2, self.card1
                self.card1_suit, self.card2_suit = self.card2_suit, self.card1_suit

        return self._get_obs(), reward, terminated, False, self._get_info()

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
        card1_str = f"{self.card1_suit}{self.card_values[self.card1]}"
        card2_str = f"{self.card2_suit}{self.card_values[self.card2]}"

        output = "\n" + "="*50 + "\n"
        output += f"DRAGON GATE - Round: {self.round}\n"
        output += f"Pot: {self.pot} | Player Money: {self.money}\n"
        output += f"Cards in deck: {len(self.deck)} | Cards in discard: {len(self.discard_pile)}\n"

        # Display the two initial cards
        output += f"\nInitial Cards: {card1_str} and {card2_str}\n"

        # Display the third card and result if available
        if hasattr(self, 'card3') and self.round != 0:
            card3_str = f"{self.card3_suit}{self.card_values[self.card3]}"
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

                value_str = self.card_values[card_value]
                card_text = self.font_medium.render(f"{card_suit}{value_str}", True, black)
                self.screen.blit(card_text, (x + card_width//2 - card_text.get_width()//2,
                                         y + card_height//2 - card_text.get_height()//2))

        # Draw game info
        info_y = card_y + card_height + 50

        # Draw pot and money
        pot_text = self.font_medium.render(f"Pot: ${self.pot}", True, gold)
        self.screen.blit(pot_text, (50, info_y))

        money_text = self.font_medium.render(f"Money: ${self.money}", True, white)
        self.screen.blit(money_text, (50, info_y + 40))

        # Draw round info
        round_text = self.font_medium.render(f"Round: {self.round}", True, white)
        self.screen.blit(round_text, (50, info_y + 80))

        # Draw card range text
        if self.card1 == self.card2:
            range_text = self.font_medium.render("Cards equal: Bet if next card will be higher or lower", True, white)
        else:
            range_text = self.font_medium.render(f"Target: Card between {self.card_values[self.card1]} and {self.card_values[self.card2]}", True, white)

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
