Rosie_agent.py — Connect-4 Tournament Agent
============================================


Algorithm
---------
Iterative-Deepening Alpha-Beta Search with a window-counting heuristic and
a self-imposed wall-clock budget. Uses the helpers provided by
environment.py (place / get_valid_moves / check_win) so the rules are
guaranteed to match the tournament environment.


Core ideas (all four taught in class)
-------------------------------------
1. Alpha-beta pruning (negamax-style, fail-soft)
2. Iterative deepening with self-imposed wall-clock budget
3. Heuristic evaluation — sliding 4-cell windows + center-column bonus
4. Move ordering — try center column first

Tactical shortcuts
------------------
- If we can win this turn, just take it.
- If opponent can win next turn, block.
- Empty board → play center (Connect-4 is a first-player win — Allis 1988,
  https://www.connectfour.net/Files/connect4.pdf)

Usage
-----
    from environment import truly_dynamic_environment
    from Rosie_agent import RosieAgent

    rosie = RosieAgent("Rosie")
    players = [
        {"algo": rosie.act, "name": "Rosie",   "player": +1, "args": {}},
        {"algo": opp_fn,    "name": "Opp",     "player": -1, "args": {}},
    ]
    result, board, _ = truly_dynamic_environment(players, size=(6,7))
    rosie.reset()      # IMPORTANT: call before every new game
