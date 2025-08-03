import mediapipe as mp
import cv2
import numpy as np

class HandDetector:
    def __init__(self, max_hands=2, detection_confidence=0.5, tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, frame, draw=True):
        """
        Detect hands in a frame and optionally draw landmarks
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(frame_rgb)
        
        landmarks_list = []
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS
                    )
                
                landmarks = []
                for lm in hand_landmarks.landmark:
                    h, w, _ = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmarks.append([cx, cy])
                landmarks_list.append(landmarks)
                
        return frame, landmarks_list