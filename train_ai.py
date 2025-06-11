import pygame
import random
import json
import os
import time
import math
import multiprocessing as mp
from opponent import get_opponent_move

# Constants
BOARD_SIZE = 3
CELL_SIZE = 100
LINE_WIDTH = 5
MARGIN = 20
GAMES_PER_ROW = 2
NUM_GAMES = 1000
VISIBLE_GAMES = min(NUM_GAMES, GAMES_PER_ROW ** 2)

WIDTH = GAMES_PER_ROW * (CELL_SIZE * BOARD_SIZE + MARGIN) + MARGIN
HEIGHT = (VISIBLE_GAMES // GAMES_PER_ROW) * (CELL_SIZE * BOARD_SIZE + MARGIN) + MARGIN

# Colors
BG_COLOR = (255, 255, 255)
LINE_COLOR = (0, 0, 0)
X_COLOR = (200, 0, 0)
O_COLOR = (0, 0, 200)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Tic-Tac-Toe AI Training")
font = pygame.font.SysFont(None, CELL_SIZE // 2)
pygame.display.set_icon(pygame.image.load('ai_logo.webp'))

if not os.path.exists("results.txt"):
    with open("results.txt", "w"): pass

class TicTacToe:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = [' '] * (BOARD_SIZE * BOARD_SIZE)
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
        row_ind = square // BOARD_SIZE
        if all(self.board[row_ind * BOARD_SIZE + i] == letter for i in range(BOARD_SIZE)):
            return True
        col_ind = square % BOARD_SIZE
        if all(self.board[col_ind + i * BOARD_SIZE] == letter for i in range(BOARD_SIZE)):
            return True
        if square % 2 == 0:
            if all(self.board[i] == letter for i in [0, 4, 8]) or all(self.board[i] == letter for i in [2, 4, 6]):
                return True
        return False

def winning_opportunity(board, letter):
    for i in range(BOARD_SIZE * BOARD_SIZE):
        if board[i] == ' ':
            board[i] = letter
            t = TicTacToe()
            t.board = board[:]
            if t.winner(i, letter):
                board[i] = ' '
                return True
            board[i] = ' '
    return False

def blocking_opportunity(board, letter):
    opponent = 'O' if letter == 'X' else 'X'
    return winning_opportunity(board, opponent)

class QLearningAgent:
    def __init__(self, name='Agent', alpha=0.5, gamma=0.9, epsilon=0.1, q_table_file=None):
        self.name = name
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table_file = q_table_file
        self.load_q_table()

    def get_state(self, board):
        return ''.join(board)

    def choose_action(self, board, available_moves):
        state = self.get_state(board)
        if state not in self.q_table or len(self.q_table[state]) == 0:
            return random.choice(available_moves)
        if random.random() < self.epsilon:
            return random.choice(available_moves)
        q_values = [self.q_table[state].get(a, 0) for a in available_moves]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(available_moves, q_values) if q == max_q]
        return random.choice(best_actions)

    def learn(self, state, action, reward, next_state, done):
        if state not in self.q_table:
            self.q_table[state] = {}
        old_value = self.q_table[state].get(action, 0)
        next_max = 0 if done or next_state not in self.q_table else max(self.q_table[next_state].values(), default=0)
        self.q_table[state][action] = old_value + self.alpha * (reward + self.gamma * next_max - old_value)

    def save_q_table(self):
        if self.q_table_file:
            try:
                with open(self.q_table_file, 'w') as f:
                    json.dump(self.q_table, f)
                print(f"{self.name} Q-table saved.")
            except Exception as e:
                print("Save failed:", e)

    def load_q_table(self):
        if self.q_table_file and os.path.exists(self.q_table_file):
            with open(self.q_table_file, 'r') as f:
                self.q_table = json.load(f)
            print(f"{self.name} Q-table loaded.")

def draw_board(game, top_left):
    x0, y0 = top_left
    pygame.draw.rect(screen, BG_COLOR, pygame.Rect(x0, y0, CELL_SIZE * BOARD_SIZE, CELL_SIZE * BOARD_SIZE))
    for i in range(1, BOARD_SIZE):
        pygame.draw.line(screen, LINE_COLOR, (x0 + i * CELL_SIZE, y0), (x0 + i * CELL_SIZE, y0 + CELL_SIZE * BOARD_SIZE), LINE_WIDTH)
        pygame.draw.line(screen, LINE_COLOR, (x0, y0 + i * CELL_SIZE), (x0 + CELL_SIZE * BOARD_SIZE, y0 + i * CELL_SIZE), LINE_WIDTH)
    for idx, val in enumerate(game.board):
        if val != ' ':
            col, row = idx % BOARD_SIZE, idx // BOARD_SIZE
            center_x = x0 + col * CELL_SIZE + CELL_SIZE // 2
            center_y = y0 + row * CELL_SIZE + CELL_SIZE // 2
            text = font.render(val, True, X_COLOR if val == 'X' else O_COLOR)
            screen.blit(text, text.get_rect(center=(center_x, center_y)))

def run_simulated_game(q_table, result_queue, epsilon):
    game = TicTacToe()
    agent = QLearningAgent(epsilon=epsilon)
    agent.q_table = q_table
    state = agent.get_state(game.board)
    prev_state = None
    prev_action = None
    player = 'X'

    while not game.game_over:
        available = game.available_moves()
        if player == 'X':
            action = agent.choose_action(game.board, available)
        else:
            action = get_opponent_move(game.board, available)
        game.make_move(action, player)
        next_state = agent.get_state(game.board)

        if player == 'X' and prev_state is not None:
            reward = 0
            if game.game_over:
                if game.current_winner == 'X':
                    reward = 1
                elif game.current_winner == 'O':
                    reward = -1
                else:
                    reward = 0.5
            elif winning_opportunity(game.board, 'X'):
                reward += 0.2
            elif blocking_opportunity(game.board, 'X'):
                reward += 0.1
            agent.learn(prev_state, prev_action, reward, next_state, game.game_over)

        if player == 'X':
            prev_state = state
            prev_action = action
        state = next_state
        player = 'O' if player == 'X' else 'X'

    result = ('X' if game.current_winner == 'X' else
              'O' if game.current_winner == 'O' else 'T')
    result_queue.put((agent.q_table, result))

# Insert into your main training loop:
def main():
    manager = mp.Manager()
    q_table = manager.dict()
    result_queue = manager.Queue()
    epsilon = 0.2
    agent = QLearningAgent(epsilon=epsilon, q_table_file='q_table.json')
    agent.load_q_table()
    for k, v in agent.q_table.items():
        q_table[k] = v

    # Visual Pygame setup
    games = [TicTacToe() for _ in range(VISIBLE_GAMES)]
    players = ['X'] * VISIBLE_GAMES
    prev_states = [None] * VISIBLE_GAMES
    prev_actions = [None] * VISIBLE_GAMES

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Tic-Tac-Toe AI Training")
    font = pygame.font.SysFont(None, CELL_SIZE // 2)
    pygame.display.set_icon(pygame.image.load('ai_logo.webp'))
    clock = pygame.time.Clock()
    running = True

    last_save = time.time()
    batch = 0
    x, o, t = 0, 0, 0

    while running:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Launch multiprocessing games
        processes = []
        for _ in range(NUM_GAMES - VISIBLE_GAMES):
            p = mp.Process(target=run_simulated_game,
                           args=(q_table, result_queue, epsilon))
            p.start()
            processes.append(p)

        for game in games:  # Run visible games sequentially
            if not game.game_over:
                player = 'X'
                state = agent.get_state(game.board)
                if player == 'X':
                    action = agent.choose_action(game.board, game.available_moves())
                else:
                    action = get_opponent_move(game.board, game.available_moves())
                game.make_move(action, player)

        # Draw visible boards
        screen.fill((180, 180, 180))
        for i in range(VISIBLE_GAMES):
            row, col = divmod(i, GAMES_PER_ROW)
            draw_board(games[i], (MARGIN + col * (CELL_SIZE * BOARD_SIZE + MARGIN),
                                  MARGIN + row * (CELL_SIZE * BOARD_SIZE + MARGIN)))
        pygame.display.flip()

        for p in processes:
            p.join()

        while not result_queue.empty():
            updated_q, result = result_queue.get()
            for state, actions in updated_q.items():
                if state not in q_table:
                    q_table[state] = actions
                else:
                    q_table[state].update(actions)
            if result == 'X':
                x += 1
            elif result == 'O':
                o += 1
            else:
                t += 1

        batch += 1
        if batch % 10 == 0:
            with open("results.txt", "a") as f:
                f.write(f"X:{x},O:{o},T:{t}\n")
            x, o, t = 0, 0, 0

        if time.time() - last_save > 120:
            agent.q_table = dict(q_table)
            agent.save_q_table()
            last_save = time.time()     

    agent.q_table = dict(q_table)
    agent.save_q_table()
    pygame.quit()

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()