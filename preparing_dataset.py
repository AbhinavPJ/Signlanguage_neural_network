import os
import random
import csv
import re
from datetime import datetime
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

def relative_to_polar(x, y):
    import math
    r = (x ** 2 + y ** 2) ** 0.5
    theta = math.atan2(y, x)
    return r, theta

def longest_prefix_no_number(filename):
    name = os.path.splitext(filename)[0]
    match = re.search(r'\d', name)
    if match:
        return name[:match.start()]
    return name

def reader(img_path):
    """Process a single image to extract hand landmarks"""
    try:
        img = cv2.imread(img_path)
        if img is None:
            return [-10] * 10, longest_prefix_no_number(img_path)
        
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        
        L = []
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                wrist_x, wrist_y = handLms.landmark[0].x, handLms.landmark[0].y
                fingertip_ids = [4, 8, 12, 16, 20]
                for id in fingertip_ids:
                    lm = handLms.landmark[id]
                    rel_x = lm.x - wrist_x
                    rel_y = lm.y - wrist_y
                    r, theta = relative_to_polar(rel_x, rel_y)
                    L.append(r)
                    L.append(theta)
                break  # Only the first hand
        
        if len(L) < 10:
            L.extend([-10] * (10 - len(L)))
        return L[:10], longest_prefix_no_number(img_path)
    except Exception:
        return [-10] * 10, longest_prefix_no_number(img_path)

MAX_FILES = 20000
LOG_FILE = "progress.log"
CSV_OUTPUT = "results.csv"

# Get and shuffle filenames
filenames = [f for f in os.listdir('.') if os.path.isfile(f)]
random.shuffle(filenames)

# Prepare logging
with open(LOG_FILE, "w") as log, open(CSV_OUTPUT, "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    header = [f"f{i+1}" for i in range(10)] + ["label"]
    writer.writerow(header)

    for idx, filename in enumerate(filenames[:MAX_FILES], 1):
        try:
            features, label = reader(filename)
            writer.writerow(features + [label])
            log.write(f"[{datetime.now()}] ({idx}/{MAX_FILES}) Processed: {filename}\n")
        except Exception as e:
            log.write(f"[{datetime.now()}] Error on {filename}: {e}\n")
