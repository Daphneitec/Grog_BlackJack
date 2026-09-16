"""Shared table state for a simplified Blackjack round."""

from __future__ import annotations

from dataclasses import dataclass, field

MAX_CARDS = 3
BUST_LIMIT = 21
USER_NAME = "You"
DEALER_NAME = "Dealer"


@dataclass
class Player:
    name: str
    is_human: bool = False
    is_dealer: bool = False
    cards: list[int] = field(default_factory=list)
    standing: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return sum(self.cards)

    @property
    def busted(self) -> bool:
        return self.total > BUST_LIMIT

    @property
    def can_draw(self) -> bool:
        return not self.standing and not self.busted and len(self.cards) < MAX_CARDS

    def receive(self, value: int) -> None:
        if len(self.cards) >= MAX_CARDS:
            raise ValueError(f"{self.name} already has {MAX_CARDS} cards.")
        self.cards.append(value)
        if self.busted or len(self.cards) >= MAX_CARDS:
            self.standing = True


@dataclass
class Table:
    players: list[Player]
    current_turn: str | None = None
    phase: str = "opening"
    log: list[str] = field(default_factory=list)

    def player(self, name: str) -> Player:
        for p in self.players:
            if p.name.casefold() == name.casefold():
                return p
        raise KeyError(f"Unknown player: {name}")

    def names(self) -> list[str]:
        return [p.name for p in self.players]

    def snapshot(self, hide_others_for: str | None = None) -> str:
        lines = ["TABLE"]
        for p in self.players:
            if hide_others_for and p.name != hide_others_for and not p.is_human:
                shown = ", ".join(str(c) for c in p.cards[:1]) or "none"
                if len(p.cards) > 1:
                    shown += " + hidden"
                total = "?"
            else:
                shown = ", ".join(str(c) for c in p.cards) or "none"
                total = str(p.total)
            flags = []
            if p.busted:
                flags.append("BUST")
            elif p.standing:
                flags.append("STAND")
            elif p.can_draw:
                flags.append("can draw")
            flag = f" ({', '.join(flags)})" if flags else ""
            role = " [dealer]" if p.is_dealer else ""
            lines.append(f"  {p.name}{role}: cards=[{shown}] total={total}{flag}")
        lines.append(f"  turn={self.current_turn or 'none'}  max_cards={MAX_CARDS}")
        return "\n".join(lines)

    def eligible_winners(self) -> list[Player]:
        alive = [p for p in self.players if not p.busted]
        if not alive:
            return []
        best = max(p.total for p in alive)
        return [p for p in alive if p.total == best]
