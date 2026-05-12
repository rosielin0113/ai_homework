import time
import numpy as np
from environment import HelperFunctions

ROWS, COLS = 6, 7
CENTER_ORDER = [3, 2, 4, 1, 5, 0, 6]
WIN_SCORE = 1_000_000


# ─── Vectorized window indexing ────────────────────────────────────────
# Precompute every 4-cell window on the 6x7 board as a pair of (row, col)
# index arrays. We can then extract all windows in a single numpy op:
#     windows = board[ROW_IDX, COL_IDX]   # shape (n_windows, 4)
def _build_window_indices():
    rows, cols = [], []
    # horizontal
    for r in range(ROWS):
        for c in range(COLS - 3):
            rows.append([r, r, r, r])
            cols.append([c, c+1, c+2, c+3])
    # vertical
    for c in range(COLS):
        for r in range(ROWS - 3):
            rows.append([r, r+1, r+2, r+3])
            cols.append([c, c, c, c])
    # diagonals
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            rows.append([r, r+1, r+2, r+3])
            cols.append([c, c+1, c+2, c+3])
            rows.append([r+3, r+2, r+1, r])
            cols.append([c, c+1, c+2, c+3])
    return np.array(rows), np.array(cols)

_ROW_IDX, _COL_IDX = _build_window_indices()


def _evaluate(board, player):
    """Vectorized window-counting heuristic.
    Sum scores over every 4-cell window + center-column control."""
    windows = board[_ROW_IDX, _COL_IDX]              # (n_windows, 4)
    me  = (windows ==  player).sum(axis=1)
    opp = (windows == -player).sum(axis=1)
    empty = 4 - me - opp

    # Score patterns (vectorized)
    score = 0
    score += int(np.sum((me  == 4)) * WIN_SCORE)
    score -= int(np.sum((opp == 4)) * WIN_SCORE)
    score += int(np.sum((me  == 3) & (empty == 1)) * 10)
    score += int(np.sum((me  == 2) & (empty == 2)) * 2)
    score -= int(np.sum((opp == 3) & (empty == 1)) * 12)   # block more eagerly
    score -= int(np.sum((opp == 2) & (empty == 2)) * 2)

    # Center-column control
    score += int((board[:, 3] ==  player).sum()) * 3
    score -= int((board[:, 3] == -player).sum()) * 3
    return score


# ─── The Agent ──────────────────────────────────────────────────────────
class _Timeout(Exception):
    pass


class RosieAgent:
    def __init__(self, name="Rosie",
                 total_budget_seconds=240.0,
                 max_move_seconds=5.0,
                 verbose=False):
        """
        Tournament defaults: 240s/game, 5s/move.

        For fast local testing:
            RosieAgent("Rosie", total_budget_seconds=30, max_move_seconds=1.5)
        """
        self.name = name
        self.total_budget = float(total_budget_seconds)
        self.max_move_seconds = float(max_move_seconds)
        self.time_used = 0.0
        self.verbose = verbose
        self._deadline = 0.0
        # Transposition table: key=(board_bytes, current) → (depth, flag, val, move)
        # flag: 0 = exact, 1 = lower bound, 2 = upper bound
        self.tt = {}
        # Killer-move heuristic: depth → column that caused a beta cutoff
        self.killer = {}

    def reset(self):
        """Reset budget AND clear caches. CALL BEFORE EACH NEW GAME."""
        self.time_used = 0.0
        self.tt.clear()
        self.killer.clear()

    def act(self, board, player):
        start = time.monotonic()
        board = np.asarray(board)

        # Empty board → center
        if (board == 0).all():
            self.time_used += time.monotonic() - start
            return 3

        moves = HelperFunctions.get_valid_moves(board)

        # Take immediate win
        for c in CENTER_ORDER:
            if c in moves and HelperFunctions.check_win(
                    HelperFunctions.place(c, board, player)) == player:
                self.time_used += time.monotonic() - start
                return c

        # Block immediate loss
        for c in CENTER_ORDER:
            if c in moves and HelperFunctions.check_win(
                    HelperFunctions.place(c, board, -player)) == -player:
                self.time_used += time.monotonic() - start
                return c

        # Per-move time budget
        time_left = max(0.2, self.total_budget - self.time_used)
        moves_played = int((board != 0).sum())
        my_moves_left = max(1, (42 - moves_played + 1) // 2)
        budget = min(time_left / my_moves_left, self.max_move_seconds)
        budget = max(budget, 0.05)
        self._deadline = start + budget

        # Iterative deepening
        best_move = 3 if 3 in moves else moves[0]
        best_val = -np.inf
        for depth in range(1, 42):
            try:
                val, m = self._alphabeta(board, depth, -np.inf, np.inf, player, player)
                if m is not None:
                    best_move, best_val = m, val
                if self.verbose:
                    print(f"  [{self.name}] depth {depth:2d}: best={best_move}  val={val:+d}")
                # Found a forced win/loss already — no point digging deeper
                if abs(val) >= WIN_SCORE - 100:
                    break
            except _Timeout:
                break

        self.time_used += time.monotonic() - start
        return best_move

    def _alphabeta(self, board, depth, alpha, beta, current, root):
        if time.monotonic() >= self._deadline:
            raise _Timeout()

        winner = HelperFunctions.check_win(board)
        if winner == root:    return  WIN_SCORE - 1, None
        if winner == -root:   return -WIN_SCORE + 1, None
        if winner == 0:       return 0, None

        moves = HelperFunctions.get_valid_moves(board)
        if not moves:
            return 0, None
        if depth == 0:
            return _evaluate(board, root), None

        # ── Transposition-table lookup ──────────────────────────────
        # Key includes `current` because the value depends on whose turn it is.
        key = (board.tobytes(), current)
        tt_move = None
        entry = self.tt.get(key)
        if entry is not None:
            tt_depth, tt_flag, tt_val, tt_move = entry
            if tt_depth >= depth:
                if tt_flag == 0:
                    return tt_val, tt_move
                elif tt_flag == 1 and tt_val > alpha:
                    alpha = tt_val
                elif tt_flag == 2 and tt_val < beta:
                    beta = tt_val
                if alpha >= beta:
                    return tt_val, tt_move

        # ── Move ordering: TT move > killer > center-out ────────────
        ordered = sorted(moves, key=lambda c: abs(c - 3))
        killer = self.killer.get(depth)
        if killer is not None and killer in ordered and killer != tt_move:
            ordered.remove(killer)
            ordered.insert(0, killer)
        if tt_move is not None and tt_move in ordered:
            ordered.remove(tt_move)
            ordered.insert(0, tt_move)

        is_max = (current == root)
        original_alpha, original_beta = alpha, beta
        best_val = -np.inf if is_max else np.inf
        best_move = ordered[0]

        for c in ordered:
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
                # Beta cutoff: remember as killer for this depth
                self.killer[depth] = c
                break

        # ── Store in transposition table ────────────────────────────
        if is_max:
            if best_val <= original_alpha:
                flag = 2   # upper bound
            elif best_val >= original_beta:
                flag = 1   # lower bound
            else:
                flag = 0   # exact
        else:
            if best_val >= original_beta:
                flag = 1
            elif best_val <= original_alpha:
                flag = 2
            else:
                flag = 0
        self.tt[key] = (depth, flag, best_val, best_move)

        return best_val, best_move
