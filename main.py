import cv2
import numpy as np
from src.hand_detector import HandDetector
from src.gesture_classifier import GestureClassifier

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Initialize detectors and classifier
    hand_detector = HandDetector()
    gesture_classifier = GestureClassifier()

    while True:
        # Read frame from webcam
        success, frame = cap.read()
        if not success:
            print("Failed to read from webcam")
            break

        # Detect hands and get landmarks
        frame, landmarks = hand_detector.find_hands(frame)

        # Classify gesture if hands are detected
        if landmarks:
            gesture, confidence = gesture_classifier.predict(landmarks)
            
            # Display gesture and confidence
            if gesture:
                cv2.putText(frame, f"Gesture: {gesture}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display FPS
        fps = cap.get(cv2.CAP_PROP_FPS)
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show frame
        cv2.imshow("Gesture Recognition", frame)

        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()