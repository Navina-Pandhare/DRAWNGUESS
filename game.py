import pygame
import numpy as np
from PIL import Image
from tensorflow import keras
import random

# ----------------- Load Model -----------------
model_path = "doodle_model.h5"  # your trained grayscale CNN
model = keras.models.load_model(model_path, compile=False)

# ----------------- Load Labels -----------------
labels_path = "models/labels.txt"
with open(labels_path, "r") as f:
    labels = [line.strip().split(" ", 1)[-1] for line in f]

# ----------------- Pygame Setup -----------------
pygame.init()
WIDTH, HEIGHT = 700, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Draw & Guess Game")
clock = pygame.time.Clock()

# Fonts
font_small = pygame.font.SysFont(None, 28)
font_big = pygame.font.SysFont(None, 40)

# Canvas
canvas_rect = pygame.Rect(50, 100, 400, 400)
canvas_surface = pygame.Surface((400, 400))
canvas_surface.fill((255, 255, 255))  # white canvas

# Brush
brush_color = (0, 0, 0)
brush_size = 8

# Buttons
clear_button = pygame.Rect(500, 180, 150, 40)

# Game State
time_left = 60
last_tick = pygame.time.get_ticks()
ai_guess = "???"
score = 0

# Current target word
target_word = random.choice(labels)

# ----------------- Helper Functions -----------------
def draw_button(text, rect, active=True):
    color = (150, 200, 150) if active else (180, 180, 180)
    pygame.draw.rect(screen, color, rect, border_radius=8)
    txt = font_small.render(text, True, (0, 0, 0))
    screen.blit(txt, (rect.x + (rect.width - txt.get_width())//2,
                      rect.y + (rect.height - txt.get_height())//2))

def live_predict():
    global ai_guess, score, target_word, time_left

    pygame.image.save(canvas_surface, "temp.png")
    img = Image.open("temp.png").convert("L").resize((128, 128))  # grayscale
    arr = np.array(img) / 255.0

    # Detect empty canvas
    if np.mean(arr) > 0.97:
        ai_guess = "Waiting..."
        return

    arr = arr.reshape(1, 128, 128, 1).astype(np.float32)
    preds = model.predict(arr, verbose=0)[0]
    class_idx = int(np.argmax(preds))
    confidence = float(preds[class_idx])

    ai_guess = f"{labels[class_idx]} ({confidence*100:.1f}%)"

    # Check if guess matches target
    if labels[class_idx].lower() == target_word.lower() and confidence > 0.6:
        score += 1
        # Reset round
        target_word = random.choice(labels)
        time_left = 60
        canvas_surface.fill((255, 255, 255))  # clear canvas


# ----------------- Main Loop -----------------
running = True
frame_count = 0

while running:
    screen.fill((230, 230, 230))

    # Event Handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if clear_button.collidepoint(event.pos):
                canvas_surface.fill((255, 255, 255))  # clear canvas

        elif event.type == pygame.MOUSEMOTION:
            if pygame.mouse.get_pressed()[0]:  # left mouse drag
                if canvas_rect.collidepoint(event.pos):
                    x, y = event.pos
                    draw_pos = (x - canvas_rect.x, y - canvas_rect.y)
                    pygame.draw.circle(canvas_surface, brush_color, draw_pos, brush_size)

    # ---------------- Live Prediction ----------------
    frame_count += 1
    if frame_count % 30 == 0:   # every 30 frames (~0.5 sec at 60 FPS)
        live_predict()

    # Timer
    now = pygame.time.get_ticks()
    if now - last_tick >= 1000:  # every 1 sec
        time_left -= 1
        last_tick = now
        live_predict()

    if time_left <= 0:
        ai_guess = "Time up!"
        time_left = 60
        target_word = random.choice(labels)
        canvas_surface.fill((255, 255, 255))

    # Draw Canvas
    screen.blit(canvas_surface, canvas_rect.topleft)

    # Draw Buttons
    draw_button("Clear", clear_button)

    # Draw Info
    timer_text = font_big.render(f"Time: {time_left}", True, (0, 0, 0))
    screen.blit(timer_text, (500, 100))

    score_text = font_big.render(f"Score: {score}", True, (0, 0, 0))
    screen.blit(score_text, (500, 50))

    target_text = font_big.render(f"Draw: {target_word}", True, (0, 0, 0))
    screen.blit(target_text, (50, 50))

    guess_text = font_big.render(f"AI Guess: {ai_guess}", True, (0, 0, 0))
    screen.blit(guess_text, (50, 520))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
