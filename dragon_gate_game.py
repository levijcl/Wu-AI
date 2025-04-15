"""
Dragon Gate Game Core

This module contains the core game logic for the Dragon Gate card game,
separated from the reinforcement learning environment.
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional

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
        # Store the current player index to ensure we update the correct player
        current_player_idx = self.current_player_idx

        # Get current player's money
        current_money = self.player_moneys[current_player_idx]

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

        # Make sure pot and money don't go below 0
        self.pot = max(0, self.pot)
        current_money = max(0, current_money)

        # Update player's money - use the stored player index to ensure we update the right player
        self.player_moneys[current_player_idx] = current_money

        # Save the third card and other info
        self.card3 = card3
        self.card3_suit = card3_suit
        self.last_bet = bet_amount
        self.last_reward = reward
        self.high_low_choice = high_low_choice
        self.last_player_idx = current_player_idx  # Save which player just acted

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

    def get_state(self):
        """Get the current game state as a dictionary"""
        return {
            "card1": self.card1,
            "card2": self.card2,
            "card1_suit": self.card1_suit,
            "card2_suit": self.card2_suit,
            "pot": self.pot,
            "player_moneys": self.player_moneys.copy(),
            "current_player_idx": self.current_player_idx,
            "round": self.round,
            "deck_size": len(self.deck),
            "discard_pile_size": len(self.discard_pile)
        }
