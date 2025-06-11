import pygame
import random
import json
import os
import multiprocessing as mp
from opponent import get_opponent_move

# === Your classes (TicTacToe, QLearningAgent, winning_opportunity, blocking_opportunity) unchanged ===
# Paste them here exactly as you provided.

# ... (Include all classes exactly as you gave before) ...

# === Helper: Run multiple training games in one process chunk ===
def train_chunk(start_game, num_games, q_table_data):
    agent = QLearningAgent()
    agent.q_table = q_table_data  # load partial q_table dict
    
    x_wins = 0
    o_wins = 0
    ties = 0

    for game_num in range(num_games):
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

    # Return results + updated q_table from this chunk
    return (x_wins, o_wins, ties, agent.q_table)

# === Main training with parallelization + progress bar ===
def train_with_progress_bar_parallel(num_games=100000, log_interval=10000, chunk_size=1000):
    pygame.init()
    WIDTH, HEIGHT = 600, 200
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Training AI Agent (Parallel)")
    font = pygame.font.SysFont("Arial", 28)
    small_font = pygame.font.SysFont("Arial", 20)
    clock = pygame.time.Clock()

    agent = QLearningAgent()
    if not os.path.exists("results.txt"):
        with open("results.txt", "w"): pass

    total_x_wins = total_o_wins = total_ties = 0
    games_done = 0

    pool = mp.Pool(mp.cpu_count())

    # We will keep a shared Q-table in main, merge deltas returned from workers
    q_table = agent.q_table

    # Precalculate how many chunks to run
    total_chunks = num_games // chunk_size
    if num_games % chunk_size != 0:
        total_chunks += 1

    # Submit all jobs upfront
    jobs = []
    for i in range(total_chunks):
        start = i * chunk_size
        size = min(chunk_size, num_games - start)
        jobs.append(pool.apply_async(train_chunk, args=(start, size, q_table)))

    running = True
    while running:
        pygame.event.pump()  # keep pygame responsive
        screen.fill((40, 40, 40))

        # Count finished jobs
        finished_jobs = [job for job in jobs if job.ready()]
        games_done = sum(min(chunk_size, num_games - i*chunk_size) for i, job in enumerate(jobs) if job.ready())

        # Aggregate results and merge Q-tables from finished jobs
        total_x_wins = 0
        total_o_wins = 0
        total_ties = 0
        merged_q_table = {}

        for job in finished_jobs:
            xw, ow, tie, qtab = job.get()
            total_x_wins += xw
            total_o_wins += ow
            total_ties += tie

            # Merge q_table results from worker
            for state, actions in qtab.items():
                if state not in merged_q_table:
                    merged_q_table[state] = {}
                for action, val in actions.items():
                    merged_q_table[state][action] = max(val, merged_q_table[state].get(action, float('-inf')))

        # Update main q_table with merged results
        q_table = merged_q_table

        # Draw progress bar
        progress = games_done / num_games
        bar_width = int(progress * (WIDTH - 100))
        pygame.draw.rect(screen, (80, 80, 80), (50, 100, WIDTH - 100, 30))
        pygame.draw.rect(screen, (50, 200, 50), (50, 100, bar_width, 30))

        text = font.render("Training in Progress (Parallel)...", True, (255, 255, 255))
        progress_text = small_font.render(f"Games done: {games_done:,} / {num_games:,}", True, (200, 200, 200))

        screen.blit(text, (50, 40))
        screen.blit(progress_text, (50, 140))
        pygame.display.flip()

        # Save results periodically (every log_interval games)
        if games_done > 0 and games_done % log_interval < chunk_size:
            with open("results.txt", "a") as f:
                f.write(f"X:{total_x_wins},O:{total_o_wins},T:{total_ties}\n")
            with open(agent.q_table_file, 'w') as f:
                json.dump(q_table, f)

        if games_done >= num_games:
            running = False

        clock.tick(10)  # reduce CPU usage while waiting

    pool.close()
    pool.join()

    print("Training complete.")
    pygame.quit()

if __name__ == "__main__":
    mp.set_start_method("spawn")
    train_with_progress_bar_parallel(num_games=100000, log_interval=10000, chunk_size=1000)
