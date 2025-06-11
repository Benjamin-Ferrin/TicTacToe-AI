import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
import os
import time

# ==== CONFIGURATION ====
RESULTS_FILE = 'results.txt'  # Format: "X:1265,O:554,T:181"
TRIGGER_THRESHOLD = 0.90
WINDOW_SIZE = 2000
UPDATE_SCRIPT = 'other_script.py'
CHECK_INTERVAL = 2  # seconds
LINE_WIDTH = 1
ROLLING_WINDOW = 10  # Rolling average window size

# ==== PARSING ====
def parse_line(line):
    parts = line.strip().split(',')
    x = int(parts[0].split(':')[1])
    o = int(parts[1].split(':')[1])
    t = int(parts[2].split(':')[1])
    return x, o, t

def parse_results():
    x_wins, o_wins, ties = [], [], []
    if not os.path.exists(RESULTS_FILE):
        return x_wins, o_wins, ties
    with open(RESULTS_FILE, 'r') as f:
        for line in f:
            try:
                x, o, t = parse_line(line)
                x_wins.append(x)
                o_wins.append(o)
                ties.append(t)
            except (IndexError, ValueError):
                continue
    return x_wins, o_wins, ties

def compute_average_x_win_rate(lines):
    total_x = total_all = 0
    for line in lines:
        x, o, t = parse_line(line)
        total = x + o + t
        total_x += x
        total_all += total
    return total_x / total_all if total_all else 0

# ==== ROLLING AVERAGE ====
def rolling_average(data, window=ROLLING_WINDOW):
    if len(data) < window:
        return data  # Not enough data to smooth
    averaged = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        window_data = data[start:i+1]
        averaged.append(sum(window_data) / len(window_data))
    return averaged

# ==== PLOTTING ====
def plot_live(ax, x_wins, o_wins, ties):
    ax.clear()
    # Multiply X-axis by 10 as requested
    games = [i * 10 for i in range(1, len(x_wins) + 1)]

    percentages_x, percentages_o, percentages_t = [], [], []
    for x, o, t in zip(x_wins, o_wins, ties):
        total = x + o + t
        if total == 0:
            percentages_x.append(0)
            percentages_o.append(0)
            percentages_t.append(0)
        else:
            percentages_x.append((x / total) * 100)
            percentages_o.append((o / total) * 100)
            percentages_t.append((t / total) * 100)

    # Apply rolling average smoothing
    percentages_x_smoothed = rolling_average(percentages_x)
    percentages_o_smoothed = rolling_average(percentages_o)
    percentages_t_smoothed = rolling_average(percentages_t)

    ax.plot(games, percentages_x_smoothed, label='X Wins % (Rolling Avg)', color='blue', linewidth=LINE_WIDTH)
    ax.plot(games, percentages_o_smoothed, label='O Wins % (Rolling Avg)', color='red', linewidth=LINE_WIDTH)
    ax.plot(games, percentages_t_smoothed, label='Ties % (Rolling Avg)', color='gray', linewidth=LINE_WIDTH)

    ax.set_title('Tic-Tac-Toe Results Live Plot (Percentages with Rolling Average)')
    ax.set_xlabel('Game Number')
    ax.set_ylabel('Percentage (%)')
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True)
    ax.figure.canvas.draw()

class MinimalToolbar(NavigationToolbar2Tk):
    toolitems = [
        ('Home', 'Reset original view', 'home', 'home'),
        ('Pan', 'Pan axes with left mouse, zoom with right', 'move', 'pan'),
        ('Zoom', 'Zoom to rectangle', 'zoom_to_rect', 'zoom'),
    ]

# ==== MAIN ====
def main():
    root = tk.Tk()
    root.wm_title("Live Tic-Tac-Toe Results")

    fig, ax = plt.subplots(figsize=(10, 6))
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    toolbar = MinimalToolbar(canvas, root)
    toolbar.update()
    canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    last_modified = 0

    def update_loop():
        nonlocal last_modified
        try:
            current_modified = os.path.getmtime(RESULTS_FILE)
            if current_modified != last_modified:
                last_modified = current_modified
                with open(RESULTS_FILE, 'r') as f:
                    all_lines = f.readlines()
                recent_lines = all_lines[-WINDOW_SIZE:]
                avg_x_win_rate = compute_average_x_win_rate(recent_lines)

                if avg_x_win_rate > TRIGGER_THRESHOLD:
                    print("Trigger condition met. Executing update script...")
                    exec(open(UPDATE_SCRIPT).read(), globals())

                x_wins, o_wins, ties = parse_results()
                if x_wins and o_wins and ties:
                    plot_live(ax, x_wins, o_wins, ties)
                    print(f"Plot updated at {time.ctime()}")
        except FileNotFoundError:
            print(f"Waiting for {RESULTS_FILE}...")

        root.after(CHECK_INTERVAL * 1000, update_loop)

    update_loop()
    root.mainloop()

if __name__ == '__main__':
    main()
