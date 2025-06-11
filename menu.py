import pygame
import sys
import subprocess

# --- CONFIG ---
BUTTON_WIDTH = 250
BUTTON_HEIGHT = 100
BUTTON_MARGIN = 30
GRID_COLS = 2
GRID_ROWS = 3
PADDING = 50
BG_COLOR = (30, 30, 30)
BUTTON_COLOR = (70, 130, 180)
BUTTON_HOVER_COLOR = (100, 160, 210)
TEXT_COLOR = (255, 255, 255)
QUIT_X_COLOR = (200, 80, 80)
QUIT_X_HOVER_COLOR = (255, 100, 100)

# --- INIT ---
pygame.init()
screen_width = GRID_COLS * (BUTTON_WIDTH + BUTTON_MARGIN) + PADDING * 2 - BUTTON_MARGIN
screen_height = GRID_ROWS * (BUTTON_HEIGHT + BUTTON_MARGIN) + PADDING * 2
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Tic-Tac-Toe: Menu")
font = pygame.font.SysFont("arial", 24)
pygame.display.set_icon(pygame.image.load('ai_logo.webp'))

# --- SCRIPTS ---
script_paths = [
    ("Train AI", "train_ai.py"),
    ("Face AI", "face_ai.py"),
    ("Face Minimax", "face_algorithm.py"),
    ("AI vs. Minimax", "ai_vs_algorithm.py"),
    ("Human vs. Human", "human_vs_human.py"),
    ("Plot Results", "plot_results.py"),
]

def draw_button(rect, text, hover):
    color = BUTTON_HOVER_COLOR if hover else BUTTON_COLOR
    pygame.draw.rect(screen, color, rect, border_radius=15)
    label = font.render(text, True, TEXT_COLOR)
    screen.blit(label, label.get_rect(center=rect.center))

def draw_quit_x(mouse_pos):
    rect = pygame.Rect(screen_width - 40, 10, 30, 30)
    color = QUIT_X_HOVER_COLOR if rect.collidepoint(mouse_pos) else QUIT_X_COLOR
    pygame.draw.rect(screen, color, rect, border_radius=8)
    label = font.render("X", True, (255, 255, 255))
    screen.blit(label, label.get_rect(center=rect.center))
    return rect

def main():
    running = True
    while running:
        screen.fill(BG_COLOR)
        mouse_pos = pygame.mouse.get_pos()

        buttons = []
        for i, (label, path) in enumerate(script_paths):
            row, col = divmod(i, GRID_COLS)
            x = PADDING + col * (BUTTON_WIDTH + BUTTON_MARGIN)
            y = PADDING + row * (BUTTON_HEIGHT + BUTTON_MARGIN)
            rect = pygame.Rect(x, y, BUTTON_WIDTH, BUTTON_HEIGHT)
            draw_button(rect, label, rect.collidepoint(mouse_pos))
            buttons.append((rect, path))

        quit_rect = draw_quit_x(mouse_pos)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if quit_rect.collidepoint(mouse_pos):
                    running = False
                for rect, path in buttons:
                    if rect.collidepoint(mouse_pos):
                        subprocess.Popen(["python", path])

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
