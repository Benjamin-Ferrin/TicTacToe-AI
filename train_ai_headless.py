import pygame
import random
import json
import os
import time
import math
from opponent import get_opponent_move

# Constants
BOARD_SIZE = 3
NUM_GAMES = 5000000  # 5 million games
LOG_INTERVAL = 1000  # Log every 1000 games

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
        self.initial_alpha = alpha  # Store initial alpha for decay
        self.alpha = alpha  # Current learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon
        self.epsilon_decay = 0.99995  # Slower decay
        self.epsilon_min = 0.001  # Lower minimum exploration
        self.q_table_file = q_table_file
        self.load_q_table()
        self.learning_count = 0

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
            
        # Q-learning update
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

    # Buffer for partial results to ensure each line in results.txt is for exactly 1000 games
    buffer_x_wins = 0
    buffer_o_wins = 0
    buffer_ties = 0
    buffer_games = 0
    LOG_LINE_SIZE = log_interval  # Each line in results.txt will represent exactly 1000 games

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

        # Train a batch of games
        batch_size = 1000  # Process games in batches for better performance
        for _ in range(batch_size):
            game = TicTacToe()
            prev_state = None
            prev_action = None

            while not game.game_over:
                state = agent.get_state(game.board)
                available = game.available_moves()
                
                # Agent always plays X
                action = agent.choose_action(game.board, available)
                game.make_move(action, 'X')
                next_state = agent.get_state(game.board)

                if prev_state is not None:
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

                prev_state = state
                prev_action = action

                # Opponent's move
                if not game.game_over:
                    opponent_action = get_opponent_move(game.board, game.available_moves())
                    game.make_move(opponent_action, 'O')

            # Update statistics
            if game.current_winner == 'X':
                total_x_wins += 1
                buffer_x_wins += 1
            elif game.current_winner == 'O':
                total_o_wins += 1
                buffer_o_wins += 1
            else:
                total_ties += 1
                buffer_ties += 1

            games_completed += 1
            games_since_last_log += 1
            buffer_games += 1
            total_learning_steps += 1

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
            f"Games: {games_completed:,}/{num_games:,} | Epsilon: {agent.epsilon:.4f}", 
            True, TEXT_COLOR
        )
        screen.blit(text, (50, 40))
        screen.blit(stats_text, (50, 140))
        pygame.display.flip()

        clock.tick(60)  # Cap at 60 FPS

    # Final save
    agent.save_q_table()
    pygame.quit()

if __name__ == "__main__":
    train_headless()