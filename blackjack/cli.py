"""Terminal UI for the multi-agent Blackjack table."""

from __future__ import annotations

import argparse
import sys

from colorama import Fore, Style, just_fix_windows_console

from blackjack.dealer import DealerAgent
from blackjack.llm import get_llm
from blackjack.players import AI_ROSTER, make_ai_players
from blackjack.state import DEALER_NAME, MAX_CARDS, USER_NAME, Player, Table

just_fix_windows_console()
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


BANNER = r"""
  ____  _            _     _            _      _    ___
 | __ )| | __ _  ___| | __(_) __ _  ___| | __ | |  |_ _|
 |  _ \| |/ _` |/ __| |/ /| |/ _` |/ __| |/ / | |   | |
 | |_) | | (_| | (__|   < | | (_| | (__|   <  | |___| |
 |____/|_|\__,_|\___|_|\_\/ |\__,_|\___|_|\_\ |_____|___|
                       |__/   AI table  -  3+ agents
"""


def c(text: str, color: str) -> str:
    return f"{color}{text}{Style.RESET_ALL}"


def print_table(table: Table) -> None:
    print()
    print(c("-" * 56, Fore.WHITE))
    for p in table.players:
        cards = "  ".join(f"[{v}]" for v in p.cards) or "(empty)"
        if p.busted:
            status = c("BUST", Fore.RED)
        elif p.standing:
            status = c("stand", Fore.YELLOW)
        else:
            status = c("playing", Fore.GREEN)
        who = c(p.name, Fore.CYAN if p.is_human else Fore.MAGENTA)
        if p.is_dealer:
            who += c("  (AI dealer)", Fore.BLUE)
        print(f"  {who:<28} {cards:<22} total={p.total:<3} {status}")
    print(c("-" * 56, Fore.WHITE))
    print()


def interpret_user(text: str) -> str:
    lowered = text.strip().casefold()
    if not lowered:
        return "empty"
    stand_words = ("stand", "stay", "hold", "pass", "i'm good", "im good", "no more", "enough")
    quit_words = ("quit", "exit", "resign")
    hit_words = ("card", "hit", "deal", "draw", "another", "hit me", "please")
    if any(w in lowered for w in quit_words):
        return "quit"
    if any(w in lowered for w in stand_words):
        return "stand"
    if any(w in lowered for w in hit_words):
        return "hit"
    return "unknown"


def declare_winner(table: Table) -> None:
    print_table(table)
    winners = table.eligible_winners()
    if not winners:
        print(c("Everyone busted. No winner this round.", Fore.RED))
        return
    if len(winners) == 1:
        w = winners[0]
        print(c(f"Winner: {w.name} with {w.total} (under 21).", Fore.GREEN + Style.BRIGHT))
    else:
        names = ", ".join(f"{w.name} ({w.total})" for w in winners)
        print(c(f"Tie at the top: {names}", Fore.YELLOW + Style.BRIGHT))
    print()
    print("Final hands:")
    for p in sorted(table.players, key=lambda x: (x.busted, -x.total)):
        mark = " <-- winner" if p in winners else ""
        print(f"  {p.name}: {p.cards} = {p.total}{mark}")


def user_turn(table: Table, dealer: DealerAgent, *, auto_stand: bool = False) -> bool:
    player = table.player(USER_NAME)
    table.current_turn = USER_NAME
    if auto_stand:
        player.standing = True
        print(c("Demo mode: you stand with the opening cards.", Fore.YELLOW))
        return True
    print(c('Your turn. Talk to the dealer naturally - e.g. "deal me the next card" or "I\'ll stand".', Fore.CYAN))
    while player.can_draw:
        print_table(table)
        try:
            uttered = input(c("You > ", Fore.CYAN)).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return False
        kind = interpret_user(uttered)
        if kind == "empty":
            continue
        if kind == "quit":
            player.standing = True
            print("You leave the hand.")
            return False
        if kind == "unknown":
            reply = dealer.ask(
                f"The human player said: {uttered!r}. "
                "If they want a card, draw one for You. If they want to stand, do not draw. "
                "Otherwise ask them to hit or stand.",
                extra_system="It is the human player's turn (name=You). Draw for You only if they clearly want a card.",
            )
            print(c("Dealer: ", Fore.BLUE) + reply)
            continue
        reply = dealer.draw_for(USER_NAME, uttered)
        print(c("Dealer: ", Fore.BLUE) + reply)
        if kind == "stand":
            player.standing = True
            break
        if player.busted:
            print(c("You busted.", Fore.RED))
            break
        if len(player.cards) >= MAX_CARDS:
            print(c("You have three cards - no more draws.", Fore.YELLOW))
            break
    player.standing = True
    return True


def run(auto_stand: bool = False) -> int:
    print(c(BANNER, Fore.GREEN))
    print("Simplified Blackjack: cards are 2-11. Highest total under 21 wins.")
    print(f"Everyone may hold at most {MAX_CARDS} cards.")
    print("Only the AI dealer can draw. Players (including you) must ask the dealer.\n")

    seated = [
        Player(USER_NAME, is_human=True),
        *[Player(name) for name, _ in AI_ROSTER],
        Player(DEALER_NAME, is_dealer=True),
    ]
    table = Table(players=seated)

    try:
        dealer_llm = get_llm(temperature=0.2)
        player_llm = get_llm(temperature=0.6)
    except Exception as exc:
        print(c(f"Could not start the model: {exc}", Fore.RED))
        return 1
    dealer = DealerAgent(table, dealer_llm)
    ai_players = make_ai_players(table, dealer, player_llm)

    print(c("Dealer is shuffling...", Fore.BLUE))
    opening_names = table.names()
    announcement = dealer.deal_opening(opening_names)
    print(c("Dealer: ", Fore.BLUE) + announcement)
    print_table(table)

    if not user_turn(table, dealer, auto_stand=auto_stand):
        print("Game ended early.")
        declare_winner(table)
        return 0

    for agent in ai_players:
        player = table.player(agent.name)
        print(c(f"\n- {agent.name}'s turn -", Fore.MAGENTA))
        while player.can_draw:
            spoken = agent.take_turn()
            print(spoken)
            print_table(table)
            if not player.can_draw:
                break
        player.standing = True

    dealer_player = table.player(DEALER_NAME)
    table.current_turn = DEALER_NAME
    print(c("\n- Dealer's hand -", Fore.BLUE))
    while dealer_player.can_draw:
        reply = dealer.ask(
            "It is your own hand now. House style: draw if your total is 16 or less, "
            "stand on 17 or more. Use the tool to draw for Dealer if you hit.",
            extra_system="This is the dealer's own turn. Draw only for Dealer.",
        )
        print(c("Dealer: ", Fore.BLUE) + reply)
        if dealer_player.total >= 17 or not dealer_player.can_draw:
            dealer_player.standing = True
        print_table(table)
        if dealer_player.standing or not dealer_player.can_draw:
            break
    dealer_player.standing = True

    print(c("\n=== Outcome ===", Fore.WHITE + Style.BRIGHT))
    declare_winner(table)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-agent simplified Blackjack")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Non-interactive: stand the human after the opening deal, then let the AI agents finish.",
    )
    args = parser.parse_args()
    raise SystemExit(run(auto_stand=args.demo))


if __name__ == "__main__":
    main()
