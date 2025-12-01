import base64
import cv2
import numpy as np
import mediapipe as mp
import json
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from tensorflow.keras.models import load_model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Load model and labels
# -----------------------
model = load_model("gesture_model.h5")
LABELS = ["hello", "thanks", "yes", "no", "sorry", 
          "i_dont_know", "help_me", "maybe","i_dont_understand",
          "finished", "what_page", "slow_down", "i_have_a_question",
          "is_this_correct", "can_you_repeat_that"]

# -----------------------
# Mediapipe setup
# -----------------------
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
hands = mp_hands.Hands(max_num_hands=2)
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

SEQUENCE_LENGTH = 30

# -----------------------
# Normalize landmarks
# -----------------------
def normalize_landmarks(hand_landmarks, pose_landmarks):
    if pose_landmarks is None:
        return None

    landmarks = []

    head = pose_landmarks[0]  # nose
    left_shoulder = pose_landmarks[11]
    right_shoulder = pose_landmarks[12]
    shoulder_dist = np.linalg.norm([
        left_shoulder[0] - right_shoulder[0],
        left_shoulder[1] - right_shoulder[1]
    ]) + 1e-6

    # Hand
    if hand_landmarks:
        for lm in hand_landmarks:
            x = (lm[0] - head[0]) / shoulder_dist
            y = (lm[1] - head[1]) / shoulder_dist
            z = lm[2] / shoulder_dist
            landmarks.extend([x, y, z])
    else:
        landmarks.extend([0]*63)

    # Pose
    for lm in pose_landmarks:
        x = (lm[0] - head[0]) / shoulder_dist
        y = (lm[1] - head[1]) / shoulder_dist
        z = lm[2] / shoulder_dist
        landmarks.extend([x, y, z])

    return landmarks

# -----------------------
# WebSocket endpoint
# -----------------------
@app.websocket("/ws/sign")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connected")

    frame_buffer = []

    while True:
        try:
            message = await websocket.receive_text()
            data = json.loads(message)
            base64_frames = data["frames"]

            # Convert all frames to normalized landmarks
            for base64_img in base64_frames:
                if not base64_img:
                    continue

                img_bytes = base64.b64decode(base64_img)
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                hand_results = hands.process(frame_rgb)
                pose_results = pose.process(frame_rgb)

                # Hand landmarks
                if hand_results.multi_hand_landmarks:
                    hand_lm = [[lm.x, lm.y, lm.z] for lm in hand_results.multi_hand_landmarks[0].landmark]
                else:
                    hand_lm = None

                # Pose landmarks
                if pose_results.pose_landmarks:
                    pose_lm = [[lm.x, lm.y, lm.z] for lm in pose_results.pose_landmarks.landmark]
                else:
                    pose_lm = None

                normalized = normalize_landmarks(hand_lm, pose_lm)
                if normalized:
                    frame_buffer.append(normalized)

            # Once we have SEQUENCE_LENGTH frames, predict
            if len(frame_buffer) >= SEQUENCE_LENGTH:
                sequence_input = np.expand_dims(frame_buffer[-SEQUENCE_LENGTH:], axis=0)  # shape: (1,30,features)
                pred = model.predict(sequence_input, verbose=0)
                predicted_label = LABELS[int(np.argmax(pred))]
                await websocket.send_text(predicted_label)

                # Optionally clear buffer or keep last frames for rolling predictions
                frame_buffer = frame_buffer[-SEQUENCE_LENGTH:]

        except Exception as e:
            print("Error:", e)
            break

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)