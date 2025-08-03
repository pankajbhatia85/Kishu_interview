import cv2
import numpy as np
import os
from hand_detector import HandDetector
import json

class DataCollector:
    def __init__(self, data_dir="data/training_data"):
        self.data_dir = data_dir
        self.hand_detector = HandDetector()
        self.gestures = ['point_left', 'point_right', 'point_up', 'point_down', 
                        'stop', 'thumbs_up', 'thumbs_down', 'pinch', 'wave']
        
        os.makedirs(data_dir, exist_ok=True)

    def collect_samples(self, gesture_name, num_samples=100):
        """
        Collect samples for a specific gesture
        """
        if gesture_name not in self.gestures:
            print(f"Invalid gesture name. Choose from: {self.gestures}")
            return

        samples = []
        cap = cv2.VideoCapture(0)
        sample_count = 0

        while sample_count < num_samples:
            success, frame = cap.read()
            if not success:
                continue

            frame, landmarks = self.hand_detector.find_hands(frame)

            if landmarks:
                cv2.putText(frame, f"Collecting {gesture_name}: {sample_count}/{num_samples}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                samples.append(landmarks[0])
                sample_count += 1

            cv2.imshow("Data Collection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        # Save samples
        save_path = os.path.join(self.data_dir, f"{gesture_name}.json")
        with open(save_path, 'w') as f:
            json.dump(samples, f)

        print(f"Collected {len(samples)} samples for {gesture_name}")

if __name__ == "__main__":
    collector = DataCollector()
    for gesture in collector.gestures:
        print(f"Collecting samples for {gesture}")
        collector.collect_samples(gesture)