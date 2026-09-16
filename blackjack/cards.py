"""Local card-drawing simulation.

Only the AI dealer agent is allowed to import and call ``draw_card``.
Player agents and the human must request cards through the dealer.
"""

from __future__ import annotations

import random


def draw_card() -> int:
    """Return a random simplified blackjack card value between 2 and 11."""
    return random.randint(2, 11)
