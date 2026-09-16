"""LangChain AI dealer — the only agent that may call draw_card."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from blackjack.cards import draw_card
from blackjack.state import MAX_CARDS, Table

DEALER_SYSTEM = """You are the Dealer in a simplified Blackjack game at a terminal table.

Rules you must enforce:
- You are the ONLY one who can draw cards. Players never draw themselves.
- A card is a random integer from 2 to 11.
- Each participant may hold at most {max_cards} cards.
- Bust is any total over 21.
- Only draw a card for the person whose turn it is, unless you are dealing the opening cards.
- When dealing the opening round, draw exactly two cards for every named player.
- After opening cards, draw at most one card per request.
- If someone is bust, already standing, or at {max_cards} cards, refuse the draw.
- Speak in a short, table-side voice. Confirm what you drew and the player's new total.

Use the draw_card_for_player tool whenever you actually deal a card.
"""


class DealerAgent:
    def __init__(self, table: Table, llm) -> None:
        self.table = table
        self.llm = llm
        self._build_tools()

    def _build_tools(self) -> None:
        table = self.table

        @tool
        def draw_card_for_player(player_name: str) -> str:
            """Draw one card (2-11) for the named player and add it to their hand.

            Only the dealer may use this tool. player_name must match a seated player.
            """
            try:
                player = table.player(player_name)
            except KeyError:
                return f"No one named {player_name} is seated. Seated: {', '.join(table.names())}"

            if len(player.cards) >= MAX_CARDS:
                return f"{player.name} already has {MAX_CARDS} cards. Cannot draw."
            if player.busted:
                return f"{player.name} already busted with {player.total}."
            if table.phase == "opening" and len(player.cards) >= 2:
                return f"{player.name} already has two opening cards."
            if table.phase != "opening":
                if not table.current_turn:
                    return "No player's turn is active. Do not draw."
                if table.current_turn.casefold() != player.name.casefold():
                    return f"It is {table.current_turn}'s turn, not {player.name}'s."
            if player.standing and table.phase != "opening":
                return f"{player.name} already stood."

            value = draw_card()
            player.receive(value)
            msg = (
                f"Dealt {value} to {player.name}. "
                f"Hand={player.cards} total={player.total}"
                f"{' BUST' if player.busted else ''}."
            )
            table.log.append(msg)
            return msg

        self.tools = [draw_card_for_player]
        self.tool_map = {t.name: t for t in self.tools}
        self.bound = self.llm.bind_tools(self.tools)

    def ask(self, user_text: str, *, extra_system: str = "") -> str:
        system = DEALER_SYSTEM.format(max_cards=MAX_CARDS)
        if extra_system:
            system += "\n" + extra_system
        system += "\n\nCurrent table:\n" + self.table.snapshot()
        messages: list = [SystemMessage(content=system), HumanMessage(content=user_text)]
        spoken = ""
        try:
            for _ in range(8):
                ai: AIMessage = self.bound.invoke(messages)
                messages.append(ai)
                if not ai.tool_calls:
                    spoken = _message_text(ai)
                    break
                for call in ai.tool_calls:
                    fn = self.tool_map[call["name"]]
                    result = fn.invoke(call["args"])
                    messages.append(
                        ToolMessage(content=str(result), tool_call_id=call["id"])
                    )
                spoken = _message_text(ai)
        except Exception as exc:
            spoken = f"(Dealer pauses: {exc})"
        if spoken:
            return spoken
        if self.table.log:
            return self.table.log[-1]
        return "The dealer nods."

    def deal_opening(self, names: list[str]) -> str:
        self.table.phase = "opening"
        drawer = self.tool_map["draw_card_for_player"]
        for name in names:
            player = self.table.player(name)
            while len(player.cards) < 2:
                drawer.invoke({"player_name": name})
        self.table.phase = "play"
        return self.ask(
            "Opening cards are already dealt using your draw tool. "
            "Do not draw any more cards. Announce each player's two-card hand and total.",
            extra_system="Do not call any tools. Only announce the current table.",
        )

    def draw_for(self, player_name: str, request_text: str) -> str:
        self.table.current_turn = player_name
        return self.ask(
            f"{player_name} says: {request_text}\n"
            f"If this is a request for a card, draw exactly one card for {player_name} "
            "using the tool. If they want to stand, draw nothing and acknowledge.",
            extra_system=(
                f"It is currently {player_name}'s turn. "
                f"Do not draw cards for anyone else."
            ),
        )


def _message_text(message: AIMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "\n".join(p for p in parts if p).strip()
    return str(content).strip()
