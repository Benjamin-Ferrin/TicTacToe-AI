import pygame
import time
import sys

# Game constants
WIDTH, HEIGHT = 300, 370
LINE_WIDTH = 5
CELL_SIZE = WIDTH // 3
FONT_SIZE = 60
BG_COLOR = (255, 255, 255)
LINE_COLOR = (0, 0, 0)
X_COLOR = (200, 0, 0)  # Human is X
O_COLOR = (0, 0, 200)  # Algorithm is O

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Tic-Tac-Toe Minimax vs. Human")
font = pygame.font.SysFont(None, FONT_SIZE)
small_font = pygame.font.SysFont(None, 28)
try:
    pygame.display.set_icon(pygame.image.load('ai_logo.ico'))
except:
    print("Warning: icon not found.")
    pygame.display.set_icon()


class TicTacToe:
    def __init__(self):
        self.board = [' '] * 9
        self.current_winner = None
        self.game_over = False

    def available_moves(self):
        return [i for i, spot in enumerate(self.board) if spot == ' ']

    def make_move(self, square, letter):
        if self.board[square] == ' ':
            self.board[square] = letter
            if self.winner(square, letter):
                self.current_winner = letter
                self.game_over = True
            elif ' ' not in self.board:
                self.game_over = True  # Tie
            return True
        return False

    def winner(self, square, letter):
        row_ind = square // 3
        if all(self.board[row_ind * 3 + i] == letter for i in range(3)):
            return True
        col_ind = square % 3
        if all(self.board[col_ind + i * 3] == letter for i in range(3)):
            return True
        if square % 2 == 0:
            if all(self.board[i] == letter for i in [0, 4, 8]) or all(self.board[i] == letter for i in [2, 4, 6]):
                return True
        return False

def is_winner(board, letter):
    wins = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # columns
        [0, 4, 8], [2, 4, 6]              # diagonals
    ]
    return any(all(board[i] == letter for i in combo) for combo in wins)

def minimax(board, player, depth=0):
    opponent = 'O' if player == 'X' else 'X'
    
    # Check for terminal states
    if is_winner(board, player):
        return (10 - depth, None)
    if is_winner(board, opponent):
        return (depth - 10, None)
    if ' ' not in board:
        return (0, None)
    
    # Initialize best score and move
    best_score = float('-inf') if player == 'O' else float('inf')
    best_move = None
    
    # Try each available move
    for move in range(9):
        if board[move] == ' ':
            board[move] = player
            score, _ = minimax(board, opponent, depth + 1)
            board[move] = ' '  # Undo move
            
            # Update best score and move
            if player == 'O':  # Maximizing player
                if score > best_score:
                    best_score = score
                    best_move = move
            else:  # Minimizing player
                if score < best_score:
                    best_score = score
                    best_move = move
    
    return (best_score, best_move)

def draw_board(board):
    screen.fill(BG_COLOR)
    for i in range(1, 3):
        pygame.draw.line(screen, LINE_COLOR, (0, CELL_SIZE * i), (WIDTH, CELL_SIZE * i), LINE_WIDTH)
        pygame.draw.line(screen, LINE_COLOR, (CELL_SIZE * i, 0), (CELL_SIZE * i, CELL_SIZE * 3), LINE_WIDTH)
    for i, val in enumerate(board):
        if val != ' ':
            x = (i % 3) * CELL_SIZE + CELL_SIZE // 2
            y = (i // 3) * CELL_SIZE + CELL_SIZE // 2
            color = X_COLOR if val == 'X' else O_COLOR
            text = font.render(val, True, color)
            text_rect = text.get_rect(center=(x, y))
            screen.blit(text, text_rect)

def draw_score(human_score, algo_score):
    human = small_font.render(f"Human (X): {human_score}", True, X_COLOR)
    algo = small_font.render(f"Algorithm (O): {algo_score}", True, O_COLOR)
    screen.blit(human, ((WIDTH / 3 - 10), HEIGHT - 60))
    screen.blit(algo, ((WIDTH / 3 - 10), HEIGHT - 30))

def get_cell_from_mouse(pos):
    x, y = pos
    if y > CELL_SIZE * 3:
        return None
    row = y // CELL_SIZE
    col = x // CELL_SIZE
    return row * 3 + col

def main():
    game = TicTacToe()
    human_score = 0
    algo_score = 0
    turn = 'X'  # Start with X
    point_awarded = False
    auto_restart = None
    game_count = 0  # Track number of games played

    clock = pygame.time.Clock()
    running = True

    while running:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if turn == 'X' and event.type == pygame.MOUSEBUTTONDOWN:
                move = get_cell_from_mouse(event.pos)
                if move is not None and game.make_move(move, 'X'):
                    turn = 'O'
                    point_awarded = False
                    auto_restart = None

        if turn == 'O' and not game.game_over:
            pygame.time.wait(300)  # Small delay for realism
            _, move = minimax(game.board[:], 'O')
            if move is not None:
                game.make_move(move, 'O')
            turn = 'X'

        if game.game_over and not point_awarded:
            if game.current_winner == 'X':
                human_score += 1
            elif game.current_winner == 'O':
                algo_score += 1
            else:
                human_score += 0.5
                algo_score += 0.5
            point_awarded = True
            auto_restart = time.time() + 2  # 2 seconds before restart
            game_count += 1

        if auto_restart and time.time() >= auto_restart:
            game = TicTacToe()
            # Alternate starting player every game
            turn = 'O' if game_count % 2 == 0 else 'X'
            auto_restart = None
            point_awarded = False

        draw_board(game.board)
        draw_score(human_score, algo_score)
        pygame.display.update()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
