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
pygame.display.set_caption("Tic-Tac-Toe Algorithm vs. Human")
font = pygame.font.SysFont(None, FONT_SIZE)
small_font = pygame.font.SysFont(None, 28)
pygame.display.set_icon(pygame.image.load('ai_logo.webp'))

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

def minimax(board, player):
    opponent = 'O' if player == 'X' else 'X'

    # 1. Win: complete a row if possible
    for i in range(9):
        if board[i] == ' ':
            board[i] = player
            if is_winner(board, player):
                board[i] = ' '
                return (1, i)
            board[i] = ' '

    # 2. Block: stop opponent’s win
    for i in range(9):
        if board[i] == ' ':
            board[i] = opponent
            if is_winner(board, opponent):
                board[i] = ' '
                return (1, i)
            board[i] = ' '

    # 3. Fork: create two threats
    def count_wins(b, p):
        cnt = 0
        for j in range(9):
            if b[j] == ' ':
                b[j] = p
                if is_winner(b, p):
                    cnt += 1
                b[j] = ' '
        return cnt

    for i in [0,2,6,8,4,1,3,5,7]:
        if board[i] == ' ':
            board[i] = player
            if count_wins(board, player) >= 2:
                board[i] = ' '
                return (1, i)
            board[i] = ' '

    # 4. Block fork
    for i in [0,2,6,8,4,1,3,5,7]:
        if board[i] == ' ':
            board[i] = opponent
            if count_wins(board, opponent) >= 2:
                board[i] = ' '
                return (1, i)
            board[i] = ' '

    # 5. Center, corners, sides order
    for i in [4, 0,2,6,8, 1,3,5,7]:
        if board[i] == ' ':
            return (0, i)

    # 6. Fallback: no moves left
    return (0, None)

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
    turn = 'X'  # Human always starts
    point_awarded = False
    auto_restart = None

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

        if auto_restart and time.time() >= auto_restart:
            game = TicTacToe()
            turn = 'X'
            auto_restart = None
            point_awarded = False

        draw_board(game.board)
        draw_score(human_score, algo_score)
        pygame.display.update()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
