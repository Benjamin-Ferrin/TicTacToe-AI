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
NUM_GAMES = 5000000  # 1 million games
LOG_INTERVAL = 1000  # Log every 1000 games
CHUNK_SIZE = 10000  # Increased chunk size for better efficiency
NUM_PROCESSES = max(1, mp.cpu_count() - 1)  # Leave one core free

# Colors for progress bar
BG_COLOR = (40, 40, 40)
BAR_COLOR = (50, 200, 50)
TEXT_COLOR = (255, 255, 255)
CLOSE_BUTTON_COLOR = (200, 50, 50)
CLOSE_BUTTON_HOVER_COLOR = (255, 70, 70)

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
    def __init__(self, name='Agent', alpha=0.3, gamma=0.95, epsilon=0.5, q_table_file='q_table.json'):
        self.name = name
        self.q_table = {}
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon
        self.epsilon_decay = 0.99995  # Slower decay
        self.epsilon_min = 0.001  # Lower minimum exploration
        self.q_table_file = q_table_file
        self.load_q_table()
        self.learning_count = 0

    def get_state(self, board):
        return ''.join(board)

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
            
        # Q-learning update with momentum
        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        self.q_table[state][action] = new_value
        
        # Increment learning count
        self.learning_count += 1
        
        # Decay epsilon based on learning count
        if self.learning_count % 100 == 0:  # Decay every 100 learning steps
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_q_table(self):
        if self.q_table_file:
            try:
                with open(self.q_table_file, 'w') as f:
                    json.dump(self.q_table, f)
                print(f"{self.name} Q-table saved. Current epsilon: {self.epsilon:.4f}")
            except Exception as e:
                print("Save failed:", e)

    def load_q_table(self):
        if self.q_table_file and os.path.exists(self.q_table_file):
            with open(self.q_table_file, 'r') as f:
                self.q_table = json.load(f)
            print(f"{self.name} Q-table loaded.")

def train_chunk(start_game, num_games, q_table_data, epsilon, epsilon_decay, epsilon_min):
    """Train a chunk of games in a separate process"""
    agent = QLearningAgent()
    agent.q_table = q_table_data.copy()  # Create a copy of the q_table
    agent.epsilon = epsilon
    agent.epsilon_decay = epsilon_decay
    agent.epsilon_min = epsilon_min
    
    x_wins = o_wins = ties = 0

    for _ in range(num_games):
        game = TicTacToe()
        player = 'X' if start_game % 2 == 0 else 'O'  # Alternate starting player
        prev_state = None
        prev_action = None

        while not game.game_over:
            state = agent.get_state(game.board)
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

        # Update statistics
        if game.current_winner == 'X':
            x_wins += 1
        elif game.current_winner == 'O':
            o_wins += 1
        else:
            ties += 1

    return (x_wins, o_wins, ties, agent.q_table, agent.epsilon, agent.learning_count)

def train_headless(num_games=NUM_GAMES, log_interval=LOG_INTERVAL):
    # Initialize pygame for progress bar
    pygame.init()
    WIDTH, HEIGHT = 800, 200
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Tic-Tac-Toe AI Training (Headless)")
    font = pygame.font.SysFont("Arial", 28)
    small_font = pygame.font.SysFont("Arial", 20)
    clock = pygame.time.Clock()
    try:
        pygame.display.set_icon(pygame.image.load('ai_logo.ico'))
    except:
        print("Warning: icon not found.")
        pygame.display.set_icon()

    # Initialize agent and results tracking
    agent = QLearningAgent()
    if not os.path.exists("results.txt"):
        with open("results.txt", "w"): pass

    total_x_wins = total_o_wins = total_ties = 0
    games_completed = 0
    last_save_time = time.time()
    total_learning_steps = 0
    games_since_last_log = 0

    # Create process pool with fixed number of processes
    pool = mp.Pool(NUM_PROCESSES)
    
    # Calculate number of chunks
    num_chunks = (num_games + CHUNK_SIZE - 1) // CHUNK_SIZE
    
    # Submit initial batch of jobs
    jobs = []
    current_epsilon = agent.epsilon
    for i in range(min(NUM_PROCESSES, num_chunks)):
        start = i * CHUNK_SIZE
        size = min(CHUNK_SIZE, num_games - start)
        jobs.append(pool.apply_async(
            train_chunk,
            args=(start, size, agent.q_table, current_epsilon, agent.epsilon_decay, agent.epsilon_min)
        ))

    next_chunk = NUM_PROCESSES
    running = True
    close_button_rect = pygame.Rect(WIDTH - 40, 10, 30, 30)
    mouse_pos = (0, 0)

    while running and games_completed < num_games:
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                agent.save_q_table()
            elif event.type == pygame.MOUSEMOTION:
                mouse_pos = event.pos
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if close_button_rect.collidepoint(mouse_pos):
                    running = False

        # Check completed jobs
        completed_jobs = [job for job in jobs if job.ready()]
        for job in completed_jobs:
            x_wins, o_wins, ties, q_table, chunk_epsilon, learning_steps = job.get()
            total_x_wins += x_wins
            total_o_wins += o_wins
            total_ties += ties
            games_completed += CHUNK_SIZE
            games_since_last_log += CHUNK_SIZE
            total_learning_steps += learning_steps

            # Merge Q-tables by averaging values
            for state, actions in q_table.items():
                if state not in agent.q_table:
                    agent.q_table[state] = actions
                else:
                    for action, value in actions.items():
                        if action not in agent.q_table[state]:
                            agent.q_table[state][action] = value
                        else:
                            # Average the Q-values
                            agent.q_table[state][action] = (agent.q_table[state][action] + value) / 2

            # Update agent's epsilon based on total learning steps
            agent.epsilon = max(agent.epsilon_min, 0.5 * (agent.epsilon_decay ** (total_learning_steps / 100)))

            # Remove completed job
            jobs.remove(job)

            # Submit new job if there are more chunks to process
            if next_chunk < num_chunks:
                start = next_chunk * CHUNK_SIZE
                size = min(CHUNK_SIZE, num_games - start)
                jobs.append(pool.apply_async(
                    train_chunk,
                    args=(start, size, agent.q_table, agent.epsilon, agent.epsilon_decay, agent.epsilon_min)
                ))
                next_chunk += 1

            # Log results when we reach the log interval
            if games_since_last_log >= log_interval:
                with open("results.txt", "a") as f:
                    f.write(f"X:{total_x_wins},O:{total_o_wins},T:{total_ties}\n")
                total_x_wins = total_o_wins = total_ties = 0
                games_since_last_log = 0

        # Auto-save every 2 minutes
        if time.time() - last_save_time >= 120:
            agent.save_q_table()
            last_save_time = time.time()

        # Update progress bar
        screen.fill(BG_COLOR)
        progress = games_completed / num_games
        bar_width = int(progress * (WIDTH - 100))
        pygame.draw.rect(screen, (80, 80, 80), (50, 100, WIDTH - 100, 30))
        pygame.draw.rect(screen, BAR_COLOR, (50, 100, bar_width, 30))

        # Draw close button
        button_color = CLOSE_BUTTON_HOVER_COLOR if close_button_rect.collidepoint(mouse_pos) else CLOSE_BUTTON_COLOR
        pygame.draw.rect(screen, button_color, close_button_rect)
        close_text = font.render("X", True, (255, 255, 255))
        close_text_rect = close_text.get_rect(center=close_button_rect.center)
        screen.blit(close_text, close_text_rect)

        # Draw text
        text = font.render(f"Training Progress: {progress*100:.1f}%", True, TEXT_COLOR)
        stats_text = small_font.render(
            f"Games: {games_completed:,}/{num_games:,} | Epsilon: {agent.epsilon:.4f} | Active Processes: {len(jobs)}", 
            True, TEXT_COLOR
        )
        screen.blit(text, (50, 40))
        screen.blit(stats_text, (50, 140))
        pygame.display.flip()

        clock.tick(60)  # Cap at 60 FPS

    # Clean up
    pool.close()
    pool.join()
    
    # Final save
    agent.save_q_table()
    pygame.quit()

if __name__ == "__main__":
    mp.set_start_method('spawn')  # Required for Windows
    train_headless()