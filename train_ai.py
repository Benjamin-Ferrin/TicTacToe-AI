import pygame
import random
import json
import os
import time
import math
from opponent import get_opponent_move
import multiprocessing
from multiprocessing import Manager, Pool, Value, Lock

# Constants
BOARD_SIZE = 3
CELL_SIZE = 100
LINE_WIDTH = 5
MARGIN = 20
GAMES_PER_ROW = 2
NUM_GAMES = 500
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
try:
    pygame.display.set_icon(pygame.image.load('ai_logo.ico'))
except:
    print("Warning: icon not found.")
    pygame.display.set_icon()

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

def count_winning_moves(board, letter):
    count = 0
    for i in range(len(board)):
        if board[i] == ' ':
            board[i] = letter
            temp = TicTacToe()
            temp.board = board[:]
            if temp.winner(i, letter):
                count += 1
            board[i] = ' '
    return count

def blocking_opportunity(board, letter):
    opponent = 'O' if letter == 'X' else 'X'
    return winning_opportunity(board, opponent)

class QLearningAgent:
    def __init__(self, name='Agent', alpha=0.3, gamma=0.95, epsilon=0.5, q_table_file=None, q_table=None, lock=None, epsilon_decay=0.99995, epsilon_min=0.001):
        self.name = name
        self.q_table = q_table if q_table is not None else {}
        self.initial_alpha = alpha  # Store initial alpha for decay
        self.alpha = alpha  # Current learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay  # Slower decay
        self.epsilon_min = epsilon_min  # Lower minimum exploration
        self.q_table_file = q_table_file
        self.lock = lock
        self.learning_count = 0
        if q_table is None:
            self.load_q_table()

    def get_state(self, board):
        # Get all possible board rotations and reflections
        states = []
        # Original board
        states.append(''.join(board))
        # Rotate 90 degrees
        rotated = [board[6], board[3], board[0], board[7], board[4], board[1], board[8], board[5], board[2]]
        states.append(''.join(rotated))
        # Rotate 180 degrees
        rotated = [board[8], board[7], board[6], board[5], board[4], board[3], board[2], board[1], board[0]]
        states.append(''.join(rotated))
        # Rotate 270 degrees
        rotated = [board[2], board[5], board[8], board[1], board[4], board[7], board[0], board[3], board[6]]
        states.append(''.join(rotated))
        # Horizontal reflection
        reflected = [board[2], board[1], board[0], board[5], board[4], board[3], board[8], board[7], board[6]]
        states.append(''.join(reflected))
        # Vertical reflection
        reflected = [board[6], board[7], board[8], board[3], board[4], board[5], board[0], board[1], board[2]]
        states.append(''.join(reflected))
        # Diagonal reflection (top-left to bottom-right)
        reflected = [board[0], board[3], board[6], board[1], board[4], board[7], board[2], board[5], board[8]]
        states.append(''.join(reflected))
        # Diagonal reflection (top-right to bottom-left)
        reflected = [board[8], board[5], board[2], board[7], board[4], board[1], board[6], board[3], board[0]]
        states.append(''.join(reflected))
        
        # Return the canonical state (lexicographically smallest)
        return min(states)

    def choose_action(self, board, available_moves):
        state = self.get_state(board)
        
        # Always take winning moves
        for move in available_moves:
            temp_board = board[:]
            temp_board[move] = 'X'
            if winning_opportunity(temp_board, 'X'):
                return move
                
        # Always block opponent's winning moves
        for move in available_moves:
            temp_board = board[:]
            temp_board[move] = 'O'
            if winning_opportunity(temp_board, 'O'):
                return move

        # Epsilon-greedy strategy
        if random.random() < self.epsilon:
            return random.choice(available_moves)
            
        if state not in self.q_table or len(self.q_table[state]) == 0:
            return random.choice(available_moves)
            
        q_values = [self.q_table[state].get(a, 0) for a in available_moves]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(available_moves, q_values) if q == max_q]
        
        return random.choice(best_actions)

    def learn(self, state, action, reward, next_state, done):
        if self.lock:
            self.lock.acquire()
        try:
            if state not in self.q_table:
                self.q_table[state] = {}
            old_value = self.q_table[state].get(action, 0)
            # Calculate next state's maximum Q-value
            if done:
                next_max = 0
            elif next_state not in self.q_table:
                next_max = 0
            else:
                next_max = max(self.q_table[next_state].values(), default=0)
            
            # Decay learning rate
            self.alpha = self.initial_alpha / (1 + self.learning_count * 0.0001)
                
            # Q-learning update with momentum
            new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
            self.q_table[state][action] = new_value
            
            # Increment learning count
            self.learning_count += 1
            
            # Decay epsilon based on learning count
            if self.learning_count % 100 == 0:  # Decay every 100 learning steps
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        finally:
            if self.lock:
                self.lock.release()

    def save_q_table(self):
        if self.q_table_file:
            try:
                with open(self.q_table_file, 'w') as f:
                    json.dump(dict(self.q_table), f)
                print(f"{self.name} Q-table saved. Current epsilon: {self.epsilon:.4f}")
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

def play_game_parallel(args):
    # This function will be run in a separate process for each game
    agent_params, q_table, lock, game_idx = args
    agent = QLearningAgent(**agent_params, q_table=q_table, lock=lock)
    game = TicTacToe()
    player = 'X' if game_idx % 2 == 0 else 'O'  # Alternate starting player
    prev_state = None
    prev_action = None
    x_win = o_win = tie = 0
    for _ in range(1):  # Play one game per call
        game.reset()
        prev_state = None
        prev_action = None
        while not game.game_over:
            state = agent.get_state(game.board)
            available = game.available_moves()
            if player == 'X':
                action = agent.choose_action(game.board, available)
            else:
                action = get_opponent_move(game.board, available)
            valid = game.make_move(action, player)
            next_state = agent.get_state(game.board)
            if player == 'X' and prev_state is not None:
                reward = 0
                if game.game_over:
                    if game.current_winner == 'X':
                        reward = 5.0  # Increased reward for winning
                    elif game.current_winner == 'O':
                        reward = -5.0  # Increased penalty for losing
                    else:
                        reward = 1.0  # Increased reward for tie
                else:
                    # Strategic rewards
                    double_threats = count_winning_moves(game.board, 'X') >= 2
                    if double_threats:
                        reward += 2.0  # Increased reward for double threats
                    elif winning_opportunity(game.board, 'X'):
                        reward += 1.0  # Increased reward for winning opportunity
                    elif blocking_opportunity(game.board, 'X'):
                        reward += 1.5  # Increased reward for blocking

                    # Position-based rewards
                    if action == 4:  # Center
                        reward += 0.3  # Increased center reward
                    elif action in [0, 2, 6, 8]:  # Corners
                        reward += 0.2  # Increased corner reward
                    elif action in [1, 3, 5, 7]:  # Edges
                        reward += 0.1  # Increased edge reward

                    # Penalize moves that give opponent winning opportunities
                    temp_board = game.board[:]
                    temp_board[action] = ' '
                    if winning_opportunity(temp_board, 'O'):
                        reward -= 1.0  # Increased penalty for giving opponent winning opportunity

                agent.learn(prev_state, prev_action, reward, next_state, game.game_over)
            if player == 'X':
                prev_state = state
                prev_action = action
            player = 'O' if player == 'X' else 'X'
        # Track results
        if game.current_winner == 'X':
            x_win = 1
        elif game.current_winner == 'O':
            o_win = 1
        else:
            tie = 1
    return (game.board, x_win, o_win, tie, agent.learning_count, agent.q_table)

def main():
    manager = Manager()
    q_table = manager.dict()
    lock = manager.Lock()
    # Use the same epsilon_decay and epsilon_min as in the headless script
    agent_params = dict(
        name='Agent',
        epsilon=0.5,
        q_table_file='q_table.json',
        alpha=0.3,
        gamma=0.95,
        epsilon_decay=0.99995,
        epsilon_min=0.001
    )
    # Load Q-table if exists
    if os.path.exists('q_table.json'):
        with open('q_table.json', 'r') as f:
            q_table.update(json.load(f))
    # For display
    display_games = [TicTacToe() for _ in range(VISIBLE_GAMES)]
    display_boards = [[' '] * (BOARD_SIZE * BOARD_SIZE) for _ in range(VISIBLE_GAMES)]
    x_wins = o_wins = ties = games_finished_in_batch = 0
    total_learning_steps = 0
    batch_size = 1000
    clock = pygame.time.Clock()
    running = True
    last_save_time = time.time()
    pool = Pool(processes=min(multiprocessing.cpu_count() - 1, NUM_GAMES))
    
    # Add close button
    close_button_rect = pygame.Rect(WIDTH - 40, 10, 30, 30)
    mouse_pos = (0, 0)
    
    # Buffer for partial results to ensure each line in results.txt is for exactly 1000 games
    buffer_x_wins = 0
    buffer_o_wins = 0
    buffer_ties = 0
    buffer_games = 0
    LOG_LINE_SIZE = 1000  # Each line in results.txt will represent exactly 1000 games
    
    while running:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                agent = QLearningAgent(**agent_params, q_table=q_table, lock=lock)
                agent.save_q_table()
            elif event.type == pygame.MOUSEMOTION:
                mouse_pos = event.pos
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if close_button_rect.collidepoint(mouse_pos):
                    running = False
                    
        if time.time() - last_save_time >= 120:
            agent = QLearningAgent(**agent_params, q_table=q_table, lock=lock)
            agent.save_q_table()
            last_save_time = time.time()
            
        screen.fill((180, 180, 180))
        # Run NUM_GAMES games in parallel
        args = [(agent_params, q_table, lock, i) for i in range(NUM_GAMES)]
        results = pool.map(play_game_parallel, args)
        # For display, update the first VISIBLE_GAMES boards
        for i in range(VISIBLE_GAMES):
            display_boards[i] = results[i][0][:]
            display_games[i].board = display_boards[i][:]
        # Aggregate results
        x_wins = sum(r[1] for r in results)
        o_wins = sum(r[2] for r in results)
        ties = sum(r[3] for r in results)
        total_learning_steps += sum(r[4] for r in results)
        games_finished_in_batch = NUM_GAMES
        
        # Merge Q-tables by averaging values with momentum
        for result in results:
            chunk_q_table = result[5]
            for state, actions in chunk_q_table.items():
                if state not in q_table:
                    q_table[state] = actions
                else:
                    for action, value in actions.items():
                        if action not in q_table[state]:
                            q_table[state][action] = value
                        else:
                            # Average the Q-values with momentum
                            q_table[state][action] = (0.7 * q_table[state][action] + 0.3 * value)
        
        # Update epsilon based on total learning steps
        agent_params['epsilon'] = max(agent_params['epsilon_min'], 
                                    0.5 * (agent_params['epsilon_decay'] ** (total_learning_steps / 100)))
        
        # Buffer the results so that each line in results.txt is for exactly 1000 games
        buffer_x_wins += x_wins
        buffer_o_wins += o_wins
        buffer_ties += ties
        buffer_games += NUM_GAMES

        # Write out as many full 1000-game lines as possible
        while buffer_games >= LOG_LINE_SIZE:
            # Calculate the proportion of results for 1000 games
            if buffer_games == LOG_LINE_SIZE:
                write_x = buffer_x_wins
                write_o = buffer_o_wins
                write_t = buffer_ties
            else:
                # Scale down the buffer to exactly 1000 games
                scale = LOG_LINE_SIZE / buffer_games
                write_x = int(round(buffer_x_wins * scale))
                write_o = int(round(buffer_o_wins * scale))
                write_t = LOG_LINE_SIZE - write_x - write_o  # Ensure sum is 1000

            with open("results.txt", "a") as f:
                f.write(f"X:{write_x},O:{write_o},T:{write_t}\n")

            # Remove the written results from the buffer
            if buffer_games == LOG_LINE_SIZE:
                buffer_x_wins = 0
                buffer_o_wins = 0
                buffer_ties = 0
                buffer_games = 0
            else:
                # Remove the written portion from the buffer
                buffer_x_wins -= write_x
                buffer_o_wins -= write_o
                buffer_ties -= write_t
                buffer_games -= LOG_LINE_SIZE
        
        # Draw close button
        button_color = (255, 70, 70) if close_button_rect.collidepoint(mouse_pos) else (200, 50, 50)
        pygame.draw.rect(screen, button_color, close_button_rect)
        close_text = font.render("X", True, (255, 255, 255))
        close_text_rect = close_text.get_rect(center=close_button_rect.center)
        screen.blit(close_text, close_text_rect)
        
        # Draw boards
        for i in range(VISIBLE_GAMES):
            row, col = divmod(i, GAMES_PER_ROW)
            draw_board(display_games[i], (MARGIN + col * (CELL_SIZE * BOARD_SIZE + MARGIN),
                                          MARGIN + row * (CELL_SIZE * BOARD_SIZE + MARGIN)))
        pygame.display.flip()
        
    # Save Q-table at the end
    agent = QLearningAgent(**agent_params, q_table=q_table, lock=lock)
    agent.save_q_table()
    pygame.quit()

if __name__ == "__main__":
    main()
