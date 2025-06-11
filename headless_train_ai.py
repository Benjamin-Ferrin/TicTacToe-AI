import random
import json
import os
import time
from opponent import get_opponent_move

class TicTacToe:
    def __init__(self):
        self.reset()

    def reset(self):
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

def winning_opportunity(board, letter):
    for i in range(9):
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
    return winning_opportunity(board, 'O' if letter == 'X' else 'X')

class QLearningAgent:
    def __init__(self, alpha=0.5, gamma=0.9, epsilon=0.2, q_table_file="q_table.json"):
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
        with open(self.q_table_file, 'w') as f:
            json.dump(self.q_table, f)

    def load_q_table(self):
        if os.path.exists(self.q_table_file):
            with open(self.q_table_file, 'r') as f:
                self.q_table = json.load(f)

def train(num_games=100000, log_interval=10000):
    agent = QLearningAgent()
    if not os.path.exists("results.txt"):
        with open("results.txt", "w"): pass

    x_wins = o_wins = ties = 0

    for game_num in range(1, num_games + 1):
        game = TicTacToe()
        player = 'X'
        prev_state = None
        prev_action = None

        while not game.game_over:
            state = agent.get_state(game.board)
            available = game.available_moves()
            action = agent.choose_action(game.board, available) if player == 'X' else get_opponent_move(game.board, available)
            game.make_move(action, player)
            next_state = agent.get_state(game.board)

            if player == 'X' and prev_state is not None:
                reward = 0
                if game.game_over:
                    reward = 1 if game.current_winner == 'X' else -1 if game.current_winner == 'O' else 0.5
                elif winning_opportunity(game.board, 'X'):
                    reward += 0.2
                elif blocking_opportunity(game.board, 'X'):
                    reward += 0.1
                agent.learn(prev_state, prev_action, reward, next_state, game.game_over)

            if player == 'X':
                prev_state = state
                prev_action = action

            player = 'O' if player == 'X' else 'X'

        if game.current_winner == 'X':
            x_wins += 1
        elif game.current_winner == 'O':
            o_wins += 1
        else:
            ties += 1

        if game_num % log_interval == 0:
            with open("results.txt", "a") as f:
                f.write(f"X:{x_wins},O:{o_wins},T:{ties}\n")
            print(f"[{game_num}] X:{x_wins}, O:{o_wins}, T:{ties}")
            x_wins = o_wins = ties = 0
            agent.save_q_table()

    agent.save_q_table()
    print("Training complete.")

if __name__ == "__main__":
    train(num_games=1000000, log_interval=100000)
