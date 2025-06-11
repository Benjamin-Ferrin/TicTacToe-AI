import random
import copy
from analyze_results import compute_average_x_win_rate

def random_opponent_move(board, available_moves):
    return random.choice(available_moves)

def medium_opponent_move(board, available_moves):
    # Take winning move if available
    for move in available_moves:
        new_board = board[:]
        new_board[move] = 'O'
        if check_winner(new_board, move, 'O'):
            return move

    # Block opponent's winning move
    for move in available_moves:
        new_board = board[:]
        new_board[move] = 'X'
        if check_winner(new_board, move, 'X'):
            return move

    # Otherwise random
    return random.choice(available_moves)

def minimax_opponent_move(board, available_moves):
    best_score = -float('inf')
    best_move = None
    for move in available_moves:
        new_board = board[:]
        new_board[move] = 'O'
        score = minimax(new_board, False)
        if score > best_score:
            best_score = score
            best_move = move
    return best_move

# --- Helper functions ---

def check_winner(board, square, letter):
    row_ind = square // 3
    if all(board[row_ind * 3 + i] == letter for i in range(3)):
        return True
    col_ind = square % 3
    if all(board[col_ind + i * 3] == letter for i in range(3)):
        return True
    if square % 2 == 0:
        if all(board[i] == letter for i in [0, 4, 8]) or all(board[i] == letter for i in [2, 4, 6]):
            return True
    return False

def minimax(board, is_maximizing):
    winner = get_winner(board)
    if winner == 'O': return 1
    if winner == 'X': return -1
    if ' ' not in board: return 0  # Tie

    if is_maximizing:
        best = -float('inf')
        for i in range(9):
            if board[i] == ' ':
                board[i] = 'O'
                val = minimax(board, False)
                board[i] = ' '
                best = max(best, val)
        return best
    else:
        best = float('inf')
        for i in range(9):
            if board[i] == ' ':
                board[i] = 'X'
                val = minimax(board, True)
                board[i] = ' '
                best = min(best, val)
        return best

def get_winner(board):
    for i in range(3):
        if board[i*3] != ' ' and board[i*3] == board[i*3+1] == board[i*3+2]:
            return board[i*3]
        if board[i] != ' ' and board[i] == board[i+3] == board[i+6]:
            return board[i]
    if board[0] != ' ' and board[0] == board[4] == board[8]:
        return board[0]
    if board[2] != ' ' and board[2] == board[4] == board[6]:
        return board[2]
    return None
