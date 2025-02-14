import cv2
import numpy as np
import pygame
import time
from cvzone.HandTrackingModule import HandDetector

# Initialize Hand Detector
detector = HandDetector(detectionCon=0.7, maxHands=1)

# Initialize Pygame
pygame.init()
WIDTH, HEIGHT = 1280, 720
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Finger Tracking Game")

# Load Target Image
target_img = pygame.image.load("chips.png")
target_size = 150
target_img = pygame.transform.scale(target_img, (target_size, target_size))

# Load Camera
cap = cv2.VideoCapture(0)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)

# Font Setup
font = pygame.font.Font(None, 50)

# Game State
game_started = False
score = 0
timer = 30
start_time = None
prev_distance = None
snap_detected = False
snap_cooldown = False  # Prevents restart during game
game_over_time = None  # Tracks when game over screen starts

# Target Position
target_x, target_y = np.random.randint(50, WIDTH - 50), np.random.randint(50, HEIGHT - 50)

def draw_text_outline(surface, text, x, y, font, color, outline_color):
    text_surface = font.render(text, True, color)
    outline_surface = font.render(text, True, outline_color)
    
    surface.blit(outline_surface, (x-2, y-2))
    surface.blit(outline_surface, (x+2, y-2))
    surface.blit(outline_surface, (x-2, y+2))
    surface.blit(outline_surface, (x+2, y+2))
    
    surface.blit(text_surface, (x, y))

running = True
while running:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Flip frame to match natural movement
    
    hands, frame = detector.findHands(frame, flipType=False)
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_surface = pygame.surfarray.make_surface(np.rot90(frame))
    frame_surface = pygame.transform.scale(frame_surface, (WIDTH, HEIGHT))
    screen.blit(frame_surface, (0, 0))
    
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]

        index_finger = lmList[8]  # Index finger tip
        thumb = lmList[4]  # Thumb tip

        ix, iy = index_finger[:2]
        tx, ty = thumb[:2]

        # Flip the x-coordinate to match Pygame
        ix = WIDTH - ix

        pygame.draw.circle(screen, (0, 255, 0), (ix, iy), 10)  # Green tracking dot

        # Calculate distance between index finger and thumb
        distance = np.linalg.norm(np.array([ix, iy]) - np.array([WIDTH - tx, ty]))

        # Snap Detection: Only trigger if game is NOT running and cooldown is over
        if not game_started and game_over_time is None and prev_distance is not None:
            if prev_distance > 50 and distance < 30:  # Thresholds for snap
                snap_detected = True

        prev_distance = distance  # Update previous distance for next frame

        if snap_detected and not game_started and game_over_time is None:
            game_started = True
            snap_cooldown = True  # Prevent further snaps from restarting game
            start_time = time.time()
            score = 0
            timer = 30
            snap_detected = False  # Reset snap detection
        
        if game_started:
            if abs(ix - target_x) < target_size // 2 and abs(iy - target_y) < target_size // 2:
                target_x, target_y = np.random.randint(50, WIDTH - 50), np.random.randint(50, HEIGHT - 50)
                score += 1

    if game_started:
        elapsed_time = time.time() - start_time
        timer = max(0, 30 - int(elapsed_time))

        screen.blit(target_img, (target_x - target_size // 2, target_y - target_size // 2))
        draw_text_outline(screen, f"Time: {timer}", 20, 20, font, (255, 255, 255), (0, 0, 0))
        draw_text_outline(screen, f"Score: {score}", 20, 70, font, (255, 255, 255), (0, 0, 0))

        if timer == 0:
            game_started = False
            snap_cooldown = True  # Prevent snap restart immediately
            game_over_time = time.time()  # Start countdown for game over screen
    
    elif game_over_time is not None:
        # Show Game Over Screen for 7 seconds before allowing restart
        elapsed_game_over = time.time() - game_over_time
        draw_text_outline(screen, "Game Over!", WIDTH // 2 - 100, HEIGHT // 2, font, (255, 255, 255), (0, 0, 0))
        draw_text_outline(screen, f"Final Score: {score}", WIDTH // 2 - 120, HEIGHT // 2 + 50, font, (255, 255, 255), (0, 0, 0))

        if elapsed_game_over >= 7:  # After 7 seconds, allow restarting again
            game_over_time = None  # Reset game over time
            snap_cooldown = False  # Allow snapping to restart
        else:
            draw_text_outline(screen, "Restarting soon...", WIDTH // 2 - 140, HEIGHT // 2 + 100, font, (255, 255, 255), (0, 0, 0))

    else:
        draw_text_outline(screen, "Snap your fingers to start!", WIDTH // 2 - 180, HEIGHT // 2, font, (255, 255, 255), (0, 0, 0))

    pygame.display.update()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

cap.release()
pygame.quit()
