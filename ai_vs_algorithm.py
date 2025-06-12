import pygame
import random
import json
import os
import sys
import time

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
pygame.display.set_caption("Tic-Tac-Toe AI vs Minimax")
font = pygame.font.SysFont(None, FONT_SIZE)
small_font = pygame.font.SysFont(None, 28)
try:
    pygame.display.set_icon(pygame.image.load('ai_logo.webp'))
except:
    print("Warning: icon not found.")


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
                self.game_over = True
            return True
        return False

    def winner(self, square, letter):
        row_ind = square // 3
        if all(self.board[row_ind*3 + i] == letter for i in range(3)):
            return True
        col_ind = square % 3
        if all(self.board[col_ind + i*3] == letter for i in range(3)):
            return True
        if square % 2 == 0:
            if all(self.board[i] == letter for i in [0,4,8]) or all(self.board[i] == letter for i in [2,4,6]):
                return True
        return False

class QLearningAgent:
    def __init__(self, name='AI', q_table_file=None):
        self.name = name
        self.q_table = {}
        if q_table_file and os.path.exists(q_table_file):
            with open(q_table_file, 'r') as f:
                self.q_table = json.load(f)

    def get_state(self, board):
        return ''.join(board)

    def choose_action(self, board, available_moves):
        state = self.get_state(board)
        if state not in self.q_table:
            return random.choice(available_moves)
        qv = self.q_table[state]
        vals = {a: qv.get(str(a), 0) for a in available_moves}
        m = max(vals.values())
        return int([a for a, v in vals.items() if v == m][0])

class PerfectAgent:
    def __init__(self, letter):
        self.letter = letter
        self.opponent = 'O' if letter == 'X' else 'X'

    def choose_action(self, board, available_moves):
        _, move = self.minimax(board[:], self.letter)
        return move

    def minimax(self, board, player):
        opponent = 'O' if player == 'X' else 'X'

        def is_winner(b, letter):
            wins = [
                [0, 1, 2], [3, 4, 5], [6, 7, 8],
                [0, 3, 6], [1, 4, 7], [2, 5, 8],
                [0, 4, 8], [2, 4, 6]
            ]
            return any(all(b[i] == letter for i in combo) for combo in wins)

        def count_wins(b, p):
            cnt = 0
            for j in range(9):
                if b[j] == ' ':
                    b[j] = p
                    if is_winner(b, p):
                        cnt += 1
                    b[j] = ' '
            return cnt

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
        for i in [0, 2, 6, 8, 4, 1, 3, 5, 7]:
            if board[i] == ' ':
                board[i] = player
                if count_wins(board, player) >= 2:
                    board[i] = ' '
                    return (1, i)
                board[i] = ' '

        # 4. Block fork
        for i in [0, 2, 6, 8, 4, 1, 3, 5, 7]:
            if board[i] == ' ':
                board[i] = opponent
                if count_wins(board, opponent) >= 2:
                    board[i] = ' '
                    return (1, i)
                board[i] = ' '

        # 5. Center, corners, sides order
        for i in [4, 0, 2, 6, 8, 1, 3, 5, 7]:
            if board[i] == ' ':
                return (0, i)

        # 6. Fallback: no moves left
        return (0, None)

def draw_board(board):
    screen.fill(BG_COLOR)
    for i in range(1,3):
        pygame.draw.line(screen, LINE_COLOR, (0,CELL_SIZE*i), (WIDTH,CELL_SIZE*i), LINE_WIDTH)
        pygame.draw.line(screen, LINE_COLOR, (CELL_SIZE*i,0), (CELL_SIZE*i,CELL_SIZE*3), LINE_WIDTH)
    for i, v in enumerate(board):
        if v != ' ':
            x = (i%3)*CELL_SIZE + CELL_SIZE//2
            y = (i//3)*CELL_SIZE + CELL_SIZE//2
            color = X_COLOR if v=='X' else O_COLOR
            txt = font.render(v, True, color)
            screen.blit(txt, txt.get_rect(center=(x,y)))

def draw_button(rect, text, mouse_pos):
    color = BUTTON_HOVER_COLOR if rect.collidepoint(mouse_pos) else BUTTON_COLOR
    pygame.draw.rect(screen, color, rect, border_radius=5)
    txt = small_font.render(text, True, BUTTON_TEXT_COLOR)
    screen.blit(txt, txt.get_rect(center=rect.center))

def draw_score(ai_score, algo_score):
    ai_lbl = small_font.render(f"AI (X): {ai_score}", True, X_COLOR)
    algo_lbl = small_font.render(f"Algo (O): {algo_score}", True, O_COLOR)
    screen.blit(ai_lbl, (10, HEIGHT-30))
    screen.blit(algo_lbl, (WIDTH-130, HEIGHT-30))

def main():
    game = TicTacToe()
    agent_X = QLearningAgent(name='AI', q_table_file='q_table_X.json')
    algo_O = PerfectAgent('O')

    restart_rect = pygame.Rect((WIDTH-110)//2, HEIGHT-70, 110, 30)
    auto_restart = None
    point_awarded = False
    ai_score = algo_score = 0
    turn = 'X'
    clock = pygame.time.Clock()

    while True:
        clock.tick(30)
        mp = pygame.mouse.get_pos()
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif e.type == pygame.MOUSEBUTTONDOWN:
                if restart_rect.collidepoint(e.pos):
                    game = TicTacToe()
                    auto_restart = None
                    point_awarded = False
                    turn = 'X'

        if not game.game_over:
            time.sleep(0.15)
            if turn == 'X':
                m = agent_X.choose_action(game.board, game.available_moves())
                game.make_move(m, 'X')
                turn = 'O'
            else:
                m = algo_O.choose_action(game.board, game.available_moves())
                game.make_move(m, 'O')
                turn = 'X'

        if game.game_over and not point_awarded:
            if game.current_winner == 'X':
                ai_score += 1
            elif game.current_winner == 'O':
                algo_score += 1
            else:
                ai_score += 0.5
                algo_score += 0.5
            point_awarded = True
            auto_restart = time.time() + 0.6  #Wait time between games

        if auto_restart and time.time() >= auto_restart:
            game = TicTacToe()
            turn = 'X'
            auto_restart = None
            point_awarded = False

        draw_board(game.board)
        draw_button(restart_rect, "Restart (R)", mp)
        draw_score(ai_score, algo_score)
        pygame.display.set_caption("Tic-Tac-Toe AI vs Algo" +
                                   (f" - Winner: {game.current_winner}" if game.game_over else ""))
        pygame.display.update()

if __name__=="__main__":
    main()
