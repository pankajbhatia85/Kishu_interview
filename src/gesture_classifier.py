import tensorflow as tf
import numpy as np

class GestureClassifier:
    def __init__(self, model_path=None):
        self.gestures = ['point_left', 'point_right', 'point_up', 'point_down', 
                        'stop', 'thumbs_up', 'thumbs_down', 'pinch', 'wave']
        
        if model_path:
            self.model = tf.keras.models.load_model(model_path)
        else:
            self.model = self._build_model()

    def _build_model(self):
        """
        Build a simple CNN model for gesture classification
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(21, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(len(self.gestures), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def preprocess_landmarks(self, landmarks):
        """
        Preprocess landmarks for model input
        """
        if not landmarks:
            return None
        
        # Take the first hand if multiple hands are detected
        landmarks = landmarks[0]
        
        # Normalize coordinates
        x_coords = [l[0] for l in landmarks]
        y_coords = [l[1] for l in landmarks]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        normalized_landmarks = []
        for landmark in landmarks:
            normalized_landmarks.append([
                (landmark[0] - x_min) / (x_max - x_min + 1e-5),
                (landmark[1] - y_min) / (y_max - y_min + 1e-5)
            ])
            
        return np.array([normalized_landmarks])

    def predict(self, landmarks):
        """
        Predict gesture from landmarks
        """
        processed_input = self.preprocess_landmarks(landmarks)
        if processed_input is None:
            return None, 0
            
        prediction = self.model.predict(processed_input, verbose=0)
        gesture_idx = np.argmax(prediction[0])
        confidence = prediction[0][gesture_idx]
        
        return self.gestures[gesture_idx], confidence