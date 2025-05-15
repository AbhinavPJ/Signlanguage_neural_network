import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import mediapipe as mp
import math
mapping = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G',
    7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N',
    14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z',
    26: 'del', 27: 'space'
}
class Model(nn.Module):
    def __init__(self, input_features=10, h1=50, h2=50, output=28):
        super().__init__()
        self.fc1 = nn.Linear(input_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, output)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x
model = Model()
model.load_state_dict(torch.load("model_weights.pt", map_location=torch.device('cpu')))
model.eval()
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.7, min_tracking_confidence=0.5)
def relative_to_polar(x, y):
    r = math.sqrt(x ** 2 + y ** 2)
    theta = math.atan2(y, x)
    return r, theta
def extract_features_from_frame(frame):
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if not results.multi_hand_landmarks:
        return None  # No hand detected
    handLms = results.multi_hand_landmarks[0]
    wrist_x, wrist_y = handLms.landmark[0].x, handLms.landmark[0].y
    fingertip_ids = [4, 8, 12, 16, 20]
    features = []
    for id in fingertip_ids:
        lm = handLms.landmark[id]
        rel_x = lm.x - wrist_x
        rel_y = lm.y - wrist_y
        r, theta = relative_to_polar(rel_x, rel_y)
        features.append(r)
        features.append(theta)
    if len(features) < 10:
        features.extend([-10] * (10 - len(features)))
    return features[:10]

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        break
    features = extract_features_from_frame(frame)
    if features is None:
        display_text = ""
    else:
        input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
        output_np = output.numpy().flatten()
        top1_idx = output_np.argmax()
        predicted_label = mapping.get(top1_idx, "")
        display_text = f"Prediction: {predicted_label}"
    cv2.putText(frame, display_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.imshow("Real-time Hand Model Output", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

