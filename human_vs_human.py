import pygame
import time
import sys

# Game constants
WIDTH, HEIGHT = 300, 380
LINE_WIDTH = 5
CELL_SIZE = WIDTH // 3
FONT_SIZE = 60
BG_COLOR = (255, 255, 255)
LINE_COLOR = (0, 0, 0)
X_COLOR = (200, 0, 0)
O_COLOR = (0, 0, 200)
BUTTON_COLOR = (100, 100, 255)
BUTTON_HOVER_COLOR = (130, 130, 255)
BUTTON_TEXT_COLOR = (255, 255, 255)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Tic-Tac-Toe: Human vs. Human")
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

    def empty_squares(self):
        return ' ' in self.board

    def make_move(self, square, letter):
        if self.board[square] == ' ':
            self.board[square] = letter
            if self.winner(square, letter):
                self.current_winner = letter
                self.game_over = True
            elif not self.empty_squares():
                self.game_over = True
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

def draw_button(rect, text, mouse_pos):
    color = BUTTON_HOVER_COLOR if rect.collidepoint(mouse_pos) else BUTTON_COLOR
    pygame.draw.rect(screen, color, rect, border_radius=5)
    txt_surf = small_font.render(text, True, BUTTON_TEXT_COLOR)
    txt_rect = txt_surf.get_rect(center=rect.center)
    screen.blit(txt_surf, txt_rect)

def draw_score_counters(x_wins, o_wins):
    x_label = small_font.render(f"X: {x_wins}", True, (0, 150, 0))
    o_label = small_font.render(f"O: {o_wins}", True, (150, 0, 0))
    screen.blit(x_label, (10, HEIGHT - 30))
    screen.blit(o_label, (WIDTH - 80, HEIGHT - 30))

def get_square_under_mouse(pos):
    x, y = pos
    if x >= WIDTH or y >= CELL_SIZE * 3:
        return None
    row = y // CELL_SIZE
    col = x // CELL_SIZE
    return row * 3 + col

def main():
    game = TicTacToe()
    restart_button_rect = pygame.Rect((WIDTH - 110) // 2, HEIGHT - 70, 110, 30)
    current_letter = 'X'
    x_wins = 0
    o_wins = 0
    auto_restart_time = None
    point_awarded = False

    running = True
    clock = pygame.time.Clock()

    while running:
        clock.tick(30)
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if restart_button_rect.collidepoint(event.pos):
                    game = TicTacToe()
                    current_letter = 'X'
                    auto_restart_time = None
                    point_awarded = False
                    pygame.display.set_caption("Tic-Tac-Toe: Human vs Human")
                elif not game.game_over:
                    square = get_square_under_mouse(event.pos)
                    if square is not None and game.make_move(square, current_letter):
                        current_letter = 'O' if current_letter == 'X' else 'X'

            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                game = TicTacToe()
                current_letter = 'X'
                auto_restart_time = None
                point_awarded = False
                pygame.display.set_caption("Tic-Tac-Toe: Human vs Human")

        if game.game_over and not point_awarded:
            if game.current_winner == 'X':
                x_wins += 1
            elif game.current_winner == 'O':
                o_wins += 1
            else:
                x_wins += 0.5
                o_wins += 0.5
            point_awarded = True
            auto_restart_time = time.time() + 1.5

        if auto_restart_time and time.time() >= auto_restart_time:
            game = TicTacToe()
            current_letter = 'X'
            auto_restart_time = None
            point_awarded = False
            pygame.display.set_caption("Tic-Tac-Toe: Human vs Human")

        draw_board(game.board)
        draw_button(restart_button_rect, "Restart (R)", mouse_pos)
        draw_score_counters(x_wins, o_wins)

        if game.game_over:
            caption = {
                'X': "X wins!",
                'O': "O wins!",
                None: "Tie!"
            }[game.current_winner]
            pygame.display.set_caption(f"Tic-Tac-Toe: {caption}")
        else:
            pygame.display.set_caption(f"Tic-Tac-Toe: {current_letter}'s turn")

        pygame.display.update()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
