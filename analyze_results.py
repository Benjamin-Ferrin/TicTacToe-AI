import os

RESULTS_FILE = 'results.txt'        # Contains lines like "X:1265,O:554,T:181"
DIFFICULTY_FILE = 'difficulty.txt'
Q_TABLE_FILE = 'q_table.json'

TRIGGER_THRESHOLD = 0.90            # For calling an external script (optional)
DIFFICULTY_THRESHOLD = 0.80         # For increasing difficulty
WINDOW_SIZE = 2000
MAX_DIFFICULTY = 2

def parse_line(line):
    parts = line.strip().split(',')
    x = int(parts[0].split(':')[1])
    o = int(parts[1].split(':')[1])
    t = int(parts[2].split(':')[1])
    return x, o, t

def compute_average_x_win_rate(lines):
    total_x = total_all = 0
    for line in lines:
        x, o, t = parse_line(line)
        total = x + o + t
        total_x += x
        total_all += total
    return total_x / total_all if total_all else 0

def load_difficulty():
    try:
        with open(DIFFICULTY_FILE, 'r') as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError):
        return 0

def save_difficulty(value):
    with open(DIFFICULTY_FILE, 'w') as f:
        f.write(str(value))

def main():
    with open(RESULTS_FILE, 'r') as f:
        all_lines = f.readlines()

    recent_lines = all_lines[-WINDOW_SIZE:]
    avg_x_win_rate = compute_average_x_win_rate(recent_lines)

    difficulty = load_difficulty()

    if not os.path.exists(Q_TABLE_FILE):
        difficulty = 0
    elif avg_x_win_rate >= DIFFICULTY_THRESHOLD:
        difficulty = min(difficulty + 1, MAX_DIFFICULTY)
    elif avg_x_win_rate <= DIFFICULTY_THRESHOLD / 4:
        difficulty = max(difficulty - 1, MAX_DIFFICULTY)

    save_difficulty(difficulty)

if __name__ == '__main__':
    main()
