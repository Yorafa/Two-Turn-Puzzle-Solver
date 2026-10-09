"""
Pathfinding and geometric connectivity checking for 2-turn LianLianKan (Shisen-Sho).
Replaces recursive DFS with exact O(ROW + COL) scanline checks.
"""


def is_line_clear(board: list[list[int]], r1: int, c1: int, r2: int, c2: int) -> bool:
    """
    Check if the straight path between (r1, c1) and (r2, c2) is completely empty.
    Endpoints (r1, c1) and (r2, c2) are excluded from the empty check.
    """
    if r1 == r2:
        if c1 == c2:
            return True
        step = 1 if c2 > c1 else -1
        for c in range(c1 + step, c2, step):
            if board[r1][c] != -1:
                return False
        return True
    elif c1 == c2:
        step = 1 if r2 > r1 else -1
        for r in range(r1 + step, r2, step):
            if board[r][c1] != -1:
                return False
        return True
    return False


def can_connect(
    board: list[list[int]],
    p1: tuple[int, int],
    p2: tuple[int, int],
    row_max: int = 6,
    col_max: int = 11,
) -> bool:
    """
    Check if two points p1 and p2 can be connected with at most 2 turns (3 segments).
    Operates in O(ROW + COL) time without recursion.
    """
    r1, c1 = p1
    r2, c2 = p2
    if p1 == p2:
        return False

    # 1. 0-turn (straight line segment)
    if (r1 == r2 or c1 == c2) and is_line_clear(board, r1, c1, r2, c2):
        return True

    # 2. 1-turn (L-shaped path via corner point)
    # Corner 1: (r1, c2)
    if board[r1][c2] == -1 and is_line_clear(board, r1, c1, r1, c2) and is_line_clear(board, r1, c2, r2, c2):
        return True
    # Corner 2: (r2, c1)
    if board[r2][c1] == -1 and is_line_clear(board, r1, c1, r2, c1) and is_line_clear(board, r2, c1, r2, c2):
        return True

    # 3. 2-turn: Horizontal ray cast across all rows r in [0, row_max + 1]
    # Check path: p1 -> (r, c1) -> (r, c2) -> p2
    for r in range(row_max + 2):
        if (
            (r == r1 or board[r][c1] == -1)
            and (r == r2 or board[r][c2] == -1)
            and is_line_clear(board, r1, c1, r, c1)
            and is_line_clear(board, r2, c2, r, c2)
            and is_line_clear(board, r, c1, r, c2)
        ):
            return True

    # 4. 2-turn: Vertical ray cast across all columns c in [0, col_max + 1]
    # Check path: p1 -> (r1, c) -> (r2, c) -> p2
    for c in range(col_max + 2):
        if (
            (c == c1 or board[r1][c] == -1)
            and (c == c2 or board[r2][c] == -1)
            and is_line_clear(board, r1, c1, r1, c)
            and is_line_clear(board, r2, c2, r2, c)
            and is_line_clear(board, r1, c, r2, c)
        ):
            return True

    return False


def get_connection_path(
    board: list[list[int]],
    p1: tuple[int, int],
    p2: tuple[int, int],
    row_max: int = 6,
    col_max: int = 11,
) -> list[tuple[int, int]] | None:
    """
    Find the actual path points connecting p1 and p2 (with <= 2 turns).
    Returns list of vertices:
      - [p1, p2] (0 turns)
      - [p1, corner, p2] (1 turn)
      - [p1, mid1, mid2, p2] (2 turns)
    or None if no path exists.
    """
    r1, c1 = p1
    r2, c2 = p2
    if p1 == p2:
        return None

    # 0-turn
    if (r1 == r2 or c1 == c2) and is_line_clear(board, r1, c1, r2, c2):
        return [p1, p2]

    # 1-turn
    if board[r1][c2] == -1 and is_line_clear(board, r1, c1, r1, c2) and is_line_clear(board, r1, c2, r2, c2):
        return [p1, (r1, c2), p2]
    if board[r2][c1] == -1 and is_line_clear(board, r1, c1, r2, c1) and is_line_clear(board, r2, c1, r2, c2):
        return [p1, (r2, c1), p2]

    # 2-turn (horizontal scan)
    for r in range(row_max + 2):
        if (
            (r == r1 or board[r][c1] == -1)
            and (r == r2 or board[r][c2] == -1)
            and is_line_clear(board, r1, c1, r, c1)
            and is_line_clear(board, r2, c2, r, c2)
            and is_line_clear(board, r, c1, r, c2)
        ):
            mid1 = (r, c1)
            mid2 = (r, c2)
            path = [p1]
            if mid1 != p1:
                path.append(mid1)
            if mid2 != mid1 and mid2 != p2:
                path.append(mid2)
            path.append(p2)
            return path

    # 2-turn (vertical scan)
    for c in range(col_max + 2):
        if (
            (c == c1 or board[r1][c] == -1)
            and (c == c2 or board[r2][c] == -1)
            and is_line_clear(board, r1, c1, r1, c)
            and is_line_clear(board, r2, c2, r2, c)
            and is_line_clear(board, r1, c, r2, c)
        ):
            mid1 = (r1, c)
            mid2 = (r2, c)
            path = [p1]
            if mid1 != p1:
                path.append(mid1)
            if mid2 != mid1 and mid2 != p2:
                path.append(mid2)
            path.append(p2)
            return path

    return None


def find_all_valid_moves(
    board: list[list[int]],
    cnter: dict[int, list[tuple[int, int]]],
    row_max: int = 6,
    col_max: int = 11,
    ignored_tiles: set[int] | None = None,
) -> list[tuple[int, tuple[int, int], tuple[int, int]]]:
    """
    Find all pairs of matching tiles that can currently connect.
    Returns list of (tile_id, pos1, pos2).
    """
    if ignored_tiles is None:
        ignored_tiles = set()

    moves = []
    for tid, positions in cnter.items():
        if tid in ignored_tiles or len(positions) < 2:
            continue
        n = len(positions)
        for i in range(n):
            for j in range(i + 1, n):
                p1, p2 = positions[i], positions[j]
                if can_connect(board, p1, p2, row_max, col_max):
                    moves.append((tid, p1, p2))
    return moves
