import time
import numpy as np
from environment import HelperFunctions

ROWS, COLS = 6, 7
CENTER_ORDER = [3, 2, 4, 1, 5, 0, 6]
WIN_SCORE = 1_000_000


# ─── Heuristic evaluation ───────────────────────────────────────────────
def _score_window(window, player):
    """Score one 4-cell window from `player`'s perspective.
    Asymmetric: opponent threats penalised harder than our threats reward,
    giving the agent a defensive bias."""
    me    = int((window ==  player).sum())
    opp   = int((window == -player).sum())
    empty = int((window ==       0).sum())
    if me == 4:                  return  WIN_SCORE
    if opp == 4:                 return -WIN_SCORE
    if me  == 3 and empty == 1:  return  10
    if me  == 2 and empty == 2:  return   2
    if opp == 3 and empty == 1:  return -12
    if opp == 2 and empty == 2:  return  -2
    return 0


def _evaluate(board, player):
    """Sum scores over every 4-cell window + small bonus for the center column."""
    score = 0
    score += int((board[:, 3] == player).sum()) * 3
    score -= int((board[:, 3] == -player).sum()) * 3
    # Horizontal
    for r in range(ROWS):
        for c in range(COLS - 3):
            score += _score_window(board[r, c:c+4], player)
    # Vertical
    for c in range(COLS):
        for r in range(ROWS - 3):
            score += _score_window(board[r:r+4, c], player)
    # Diagonals
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            score += _score_window(np.array([board[r+i,   c+i] for i in range(4)]), player)
            score += _score_window(np.array([board[r+3-i, c+i] for i in range(4)]), player)
    return score


# ─── The Agent ──────────────────────────────────────────────────────────
class _Timeout(Exception):
    """Raised inside the search to abort gracefully when the deadline is hit."""


class RosieAgent:
    def __init__(self, name="Rosie", total_budget_seconds=290.0, verbose=False):
        """
        Parameters
        ----------
        name : str
            Display name for the environment.
        total_budget_seconds : float
            Wall-clock budget for one whole game. 290 s leaves a 10-second
            safety margin under the 5-minute tournament rule.
        verbose : bool
            Print per-move depth + timing.
        """
        self.name = name
        self.total_budget = float(total_budget_seconds)
        self.time_used = 0.0
        self.verbose = verbose
        self._deadline = 0.0

    # -------------------------------------------------------------------
    def reset(self):
        """Reset the time budget. CALL BEFORE EACH NEW GAME."""
        self.time_used = 0.0

    # -------------------------------------------------------------------
    def act(self, board, player):
        """Environment entry point. Returns a column index 0..6."""
        start = time.monotonic()
        board = np.asarray(board)

        # Hardcoded opening: empty board → play center
        if (board == 0).all():
            self.time_used += time.monotonic() - start
            return 3

        moves = HelperFunctions.get_valid_moves(board)

        # Tactical shortcut 1 — immediate win
        for c in CENTER_ORDER:
            if c in moves and HelperFunctions.check_win(
                    HelperFunctions.place(c, board, player)) == player:
                self.time_used += time.monotonic() - start
                return c

        # Tactical shortcut 2 — must block
        for c in CENTER_ORDER:
            if c in moves and HelperFunctions.check_win(
                    HelperFunctions.place(c, board, -player)) == -player:
                self.time_used += time.monotonic() - start
                return c

        # Time budget for THIS move
        time_left = max(0.5, self.total_budget - self.time_used)
        moves_played = int((board != 0).sum())
        my_moves_left = max(1, (42 - moves_played + 1) // 2)
        # Spend up to 1.5x the average per-move slice; cap at 25% of budget
        # so a single slow move can never blow the whole game.
        budget = min(0.25 * time_left, 1.5 * time_left / my_moves_left)
        budget = max(budget, 0.10)
        self._deadline = start + budget

        # Iterative deepening
        best_move = 3 if 3 in moves else moves[0]
        for depth in range(1, 42):
            try:
                _, m = self._alphabeta(board, depth, -np.inf, np.inf, player, player)
                if m is not None:
                    best_move = m
                if self.verbose:
                    print(f"  [{self.name}] depth {depth:2d}: best={best_move}")
            except _Timeout:
                if self.verbose:
                    print(f"  [{self.name}] depth {depth} aborted (time)")
                break

        self.time_used += time.monotonic() - start
        return best_move

    # -------------------------------------------------------------------
    def _alphabeta(self, board, depth, alpha, beta, current, root):
        """Standard alpha-beta search.
        - `current` = player to move at this node
        - `root`    = the player WE are playing for (always evaluate from
                       root's POV so the value is the same sign at every node)"""
        if time.monotonic() >= self._deadline:
            raise _Timeout()

        winner = HelperFunctions.check_win(board)
        if winner == root:    return  WIN_SCORE - 1, None
        if winner == -root:   return -WIN_SCORE + 1, None
        if winner == 0:       return 0, None              # draw

        moves = HelperFunctions.get_valid_moves(board)
        if not moves:
            return 0, None
        if depth == 0:
            return _evaluate(board, root), None

        # Center-first move ordering
        moves.sort(key=lambda c: abs(c - 3))

        is_max = (current == root)
        best_val = -np.inf if is_max else np.inf
        best_move = moves[0]

        for c in moves:
            child = HelperFunctions.place(c, board, current)
            val, _ = self._alphabeta(child, depth - 1, alpha, beta, -current, root)
            if is_max:
                if val > best_val:
                    best_val = val
                    best_move = c
                if val > alpha:
                    alpha = val
            else:
                if val < best_val:
                    best_val = val
                    best_move = c
                if val < beta:
                    beta = val
            if alpha >= beta:
                break

        return best_val, best_move
