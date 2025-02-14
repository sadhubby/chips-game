import cv2
import mediapipe as mp
import numpy as np
import pygame
import time


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)


pygame.init()
WIDTH, HEIGHT = 1280, 720
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Finger Tracking Game")

target_img = pygame.image.load("C:/Users/Evan/Desktop/chips game/chips.png")  
target_size = 150
target_img = pygame.transform.scale(target_img, (target_size, target_size))


cap = cv2.VideoCapture(0)

font = pygame.font.Font(None, 50)  

game_started = False
snap_detected = False
score = 0
timer = 30  
start_time = None  


target_x, target_y = np.random.randint(50, WIDTH-50), np.random.randint(50, HEIGHT-50)


def draw_text_outline(surface, text, x, y, font, color, outline_color):
    text_surface = font.render(text, True, color)
    outline_surface = font.render(text, True, outline_color)
    

    surface.blit(outline_surface, (x-2, y-2))
    surface.blit(outline_surface, (x+2, y-2))
    surface.blit(outline_surface, (x-2, y+2))
    surface.blit(outline_surface, (x+2, y+2))
    
    # Draw main text
    surface.blit(text_surface, (x, y))
n
def detect_snap(hand_landmarks):
    global snap_detected

    thumb_tip = hand_landmarks.landmark[4]
    middle_tip = hand_landmarks.landmark[12]

    distance = np.linalg.norm(
        np.array([thumb_tip.x, thumb_tip.y]) - np.array([middle_tip.x, middle_tip.y])
    )

    if distance < 0.05:
        snap_detected = True
    elif snap_detected and distance > 0.1:  
        snap_detected = False
        return True  
    
    return False  

running = True
while running:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame_surface = pygame.surfarray.make_surface(np.rot90(frame))
    frame_surface = pygame.transform.scale(frame_surface, (WIDTH, HEIGHT)) 

    screen.blit(frame_surface, (0, 0))

    frame_for_mediapipe = np.flip(frame, axis=1)  


    results = hands.process(frame_for_mediapipe)

    hand_detected = False 

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            hand_detected = True  
            if detect_snap(hand_landmarks): 
                if not game_started:
                    game_started = True
                    start_time = time.time()  
                    score = 0  
                    timer = 30 

            if game_started:  
                
                index_finger = hand_landmarks.landmark[8]
                ix, iy = int(index_finger.x * WIDTH), int(index_finger.y * HEIGHT)

                pygame.draw.circle(screen, (0, 255, 0), (ix, iy), 10)

        
                if abs(ix - target_x) < target_size//2 and abs(iy - target_y) < target_size//2:
                    target_x, target_y = np.random.randint(50, WIDTH-50), np.random.randint(50, HEIGHT-50)  # New target
                    score += 1  


            for lm in hand_landmarks.landmark:
                px, py = int(lm.x * WIDTH), int(lm.y * HEIGHT)
                pygame.draw.circle(screen, (255, 0, 0), (px, py), 2)

    if game_started:
    
        elapsed_time = time.time() - start_time
        timer = max(0, 30 - int(elapsed_time))  # Countdown

    
        screen.blit(target_img, (target_x - target_size//2, target_y - target_size//2))

    
        draw_text_outline(screen, f"Time: {timer}", 20, 20, font, (255, 255, 255), (0, 0, 0))
        draw_text_outline(screen, f"Score: {score}", 20, 70, font, (255, 255, 255), (0, 0, 0))

    
        if timer == 0:
            game_started = False  
            draw_text_outline(screen, "Game Over!", WIDTH//2 - 100, HEIGHT//2, font, (255, 255, 255), (0, 0, 0))
            draw_text_outline(screen, f"Final Score: {score}", WIDTH//2 - 120, HEIGHT//2 + 50, font, (255, 255, 255), (0, 0, 0))
            draw_text_outline(screen, "Snap to Play Again!", WIDTH//2 - 150, HEIGHT//2 + 100, font, (255, 255, 255), (0, 0, 0))

    else:
        draw_text_outline(screen, "Snap to Start!", WIDTH//2 - 120, HEIGHT//2, font, (255, 255, 255), (0, 0, 0))

    pygame.display.update()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

cap.release()
pygame.quit()
