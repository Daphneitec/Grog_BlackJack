"""LangChain AI player agents. They cannot import or call draw_card."""

from __future__ import annotations

from pydantic import BaseModel, Field

from blackjack.dealer import DealerAgent
from blackjack.state import BUST_LIMIT, MAX_CARDS, Player, Table

PLAYER_SYSTEM = """You are {name}, an AI Blackjack player sitting with friends at a terminal table.

House rules:
- You may hold at most {max_cards} cards.
- Highest total that is still {bust} or under wins.
- You CANNOT draw a card yourself. The only way to get a card is to call ask_dealer_for_card.
- Call stand_pat when you do not want another card.
- Play a little personality: {persona}
- Be reasonably smart: hit on low totals, be careful near 17+, never hit if you would exceed {max_cards} cards.

Current public table:
{table}

Your private hand: {hand} (total {total}). Cards held: {count}/{max_cards}.
Decide whether to ask the dealer for another card or stand.
"""


class PlayerIntent(BaseModel):
    want_card: bool = Field(description="True if you want the dealer to deal you another card.")
    table_talk: str = Field(description="One short spoken line to the table.")


class PlayerAgent:
    def __init__(self, name: str, persona: str, table: Table, dealer: DealerAgent, llm) -> None:
        self.name = name
        self.persona = persona
        self.table = table
        self.dealer = dealer
        self.llm = llm.with_structured_output(PlayerIntent, method="json_schema", strict=True)

        # Tools exist so players must request cards through the dealer — never cards.draw_card.
        self._bind_request_tools()

    def _bind_request_tools(self) -> None:
        from langchain_core.tools import tool

        dealer = self.dealer
        name = self.name

        @tool
        def ask_dealer_for_card(message_to_dealer: str) -> str:
            """Ask the AI dealer to draw a card for you. You cannot draw cards yourself."""
            return dealer.draw_for(name, message_to_dealer)

        @tool
        def stand_pat(message_to_table: str) -> str:
            """Stay with your current hand. No card will be drawn."""
            player = dealer.table.player(name)
            player.standing = True
            note = f"{name} stands with {player.cards} (total {player.total}). {message_to_table}"
            dealer.table.log.append(note)
            return note

        self.request_card = ask_dealer_for_card
        self.stand = stand_pat

    def take_turn(self) -> str:
        player = self.table.player(self.name)
        self.table.current_turn = self.name
        if not player.can_draw:
            player.standing = True
            return f"{self.name} cannot draw and stands with {player.total}."

        prompt = PLAYER_SYSTEM.format(
            name=self.name,
            persona=self.persona,
            max_cards=MAX_CARDS,
            bust=BUST_LIMIT,
            table=self.table.snapshot(hide_others_for=self.name),
            hand=player.cards,
            total=player.total,
            count=len(player.cards),
        )
        try:
            intent: PlayerIntent = self.llm.invoke(prompt)
        except Exception:
            intent = PlayerIntent(
                want_card=player.total <= 16 and player.can_draw,
                table_talk="I'll play the percentages.",
            )

        if intent.want_card and player.can_draw:
            result = self.request_card.invoke(
                {"message_to_dealer": f"Please deal me another card. {intent.table_talk}"}
            )
            return f"{self.name}: {intent.table_talk}\nDealer: {result}"

        result = self.stand.invoke({"message_to_table": intent.table_talk})
        return f"{self.name}: {intent.table_talk}\n{result}"


AI_ROSTER: list[tuple[str, str]] = [
    ("Maya", "confident card-counter vibe, a bit cocky"),
    ("Jules", "cautious and polite, hates going bust"),
    ("Rin", "chaotic and lucky, sometimes hits on 16 just for fun"),
]


def make_ai_players(table: Table, dealer: DealerAgent, llm) -> list[PlayerAgent]:
    agents = []
    for name, persona in AI_ROSTER:
        if any(p.name == name for p in table.players):
            agents.append(PlayerAgent(name, persona, table, dealer, llm))
    return agents
