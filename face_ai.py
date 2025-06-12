import pygame
import time
import random
import json
import os
import sys
import time

# Game constants
WIDTH, HEIGHT = 300, 380  # Extra height for button and score
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
try:
    pygame.display.set_icon(pygame.image.load('ai_logo.ico'))
except:
    print("Warning: icon not found.")
    pygame.display.set_icon()

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Tic-Tac-Toe AI vs. Human")
font = pygame.font.SysFont(None, FONT_SIZE)
small_font = pygame.font.SysFont(None, 28)


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

class QLearningAgent:
    def __init__(self, name='Agent', q_table_file=None):
        self.name = name
        self.q_table = {}
        if q_table_file and os.path.exists(q_table_file):
            with open(q_table_file, 'r') as f:
                self.q_table = json.load(f)
            print(f"{self.name} loaded Q-table with {len(self.q_table)} states.")

    def get_state(self, board):
        return ''.join(board)

    def choose_action(self, board, available_moves):
        state = self.get_state(board)
        if state not in self.q_table:
            return random.choice(available_moves)
        q_values = self.q_table[state]
        filtered = {a: q_values.get(str(a), 0) for a in available_moves}
        max_q = max(filtered.values())
        best_actions = [int(a) for a, q in filtered.items() if q == max_q]
        return random.choice(best_actions)

def draw_board(board):
    screen.fill(BG_COLOR)
    # Draw grid lines
    for i in range(1, 3):
        pygame.draw.line(screen, LINE_COLOR, (0, CELL_SIZE * i), (WIDTH, CELL_SIZE * i), LINE_WIDTH)
        pygame.draw.line(screen, LINE_COLOR, (CELL_SIZE * i, 0), (CELL_SIZE * i, CELL_SIZE * 3), LINE_WIDTH)

    # Draw X and O
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

def draw_score_counters(human, ai):
    human_label = small_font.render(f"Human: {human}", True, (0, 150, 0))
    ai_label = small_font.render(f"AI: {ai}", True, (150, 0, 0))
    screen.blit(human_label, (10, HEIGHT - 30))
    screen.blit(ai_label, (WIDTH - 90, HEIGHT - 30))

def get_square_under_mouse(pos):
    x, y = pos
    if x >= WIDTH or y >= CELL_SIZE * 3:
        return None
    row = y // CELL_SIZE
    col = x // CELL_SIZE
    return row * 3 + col

def main():
    game = TicTacToe()
    agent_O = QLearningAgent(name='Agent_O', q_table_file='q_table_O.json')

    # Start with human (X); this will toggle each reset
    human_starts = True

    player_turn = human_starts  # True = human, False = AI
    restart_button_rect = pygame.Rect((WIDTH - 110) // 2, HEIGHT - 70, 110, 30)
    auto_restart_time = None
    point_awarded = False
    human_wins = 0
    ai_wins = 0

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
                    human_starts = not human_starts
                    player_turn = human_starts
                    auto_restart_time = None
                    point_awarded = False
                    pygame.display.set_caption("Tic-Tac-Toe Human vs AI")
                
                elif player_turn and not game.game_over:
                    square = get_square_under_mouse(event.pos)
                    if square is not None and game.make_move(square, 'X'):
                        player_turn = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    game = TicTacToe()
                    human_starts = not human_starts
                    player_turn = human_starts
                    auto_restart_time = None
                    point_awarded = False
                    pygame.display.set_caption("Tic-Tac-Toe Human vs AI")

        if not player_turn and not game.game_over:
            time.sleep(0.15)
            move = agent_O.choose_action(game.board, game.available_moves())
            game.make_move(move, 'O')
            player_turn = True

        elif player_turn and not game.game_over:
            # Human moves via click handled in event loop above
            pass

        # Win/tie logic
        if game.game_over and not point_awarded:
            if game.current_winner == 'X':
                human_wins += 1
            elif game.current_winner == 'O':
                ai_wins += 1
            else:
                human_wins += 0.5
                ai_wins += 0.5
            point_awarded = True
            auto_restart_time = time.time() + 1

        # Auto-reset when due
        if auto_restart_time and time.time() >= auto_restart_time:
            game = TicTacToe()
            human_starts = not human_starts
            player_turn = human_starts
            auto_restart_time = None
            point_awarded = False
            pygame.display.set_caption("Tic-Tac-Toe Human vs AI")

        draw_board(game.board)
        draw_button(restart_button_rect, "Restart (R)", mouse_pos)
        draw_score_counters(human_wins, ai_wins)

        if game.game_over:
            caption = {
              'X': "You win!",
              'O': "AI wins!",
              None: "Tie!"
            }[game.current_winner]
            pygame.display.set_caption(f"Tic-Tac-Toe Human vs AI - {caption}")
        else:
            pygame.display.set_caption("Tic-Tac-Toe Human vs AI")

        pygame.display.update()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
