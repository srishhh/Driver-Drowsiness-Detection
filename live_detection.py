import os
import cv2
import numpy as np
import pygame
from tensorflow.keras.models import load_model

# --------------------------------------------------
# 1. Load the trained CNN model
# --------------------------------------------------

MODEL_PATH = "best_model.h5"
ALARM_PATH = "alarm.wav"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}. "
        "Run the CNN training notebook first."
    )

model = load_model(MODEL_PATH, compile=False)

# --------------------------------------------------
# 2. Load OpenCV Haar cascade classifiers
# --------------------------------------------------

cascade_dir = cv2.data.haarcascades

face_cascade = cv2.CascadeClassifier(
    os.path.join(cascade_dir, "haarcascade_frontalface_default.xml")
)

eye_cascade = cv2.CascadeClassifier(
    os.path.join(cascade_dir, "haarcascade_eye.xml")
)

if face_cascade.empty() or eye_cascade.empty():
    raise RuntimeError("Could not load the Haar cascade classifiers.")

# --------------------------------------------------
# 3. Initialize the alarm
# --------------------------------------------------

alarm = None
alarm_channel = None

if os.path.exists(ALARM_PATH):
    try:
        pygame.mixer.init()
        alarm = pygame.mixer.Sound(ALARM_PATH)
    except pygame.error as error:
        print(f"Audio initialization failed: {error}")
else:
    print("alarm.wav not found. Detection will run without sound.")

# --------------------------------------------------
# 4. Initialize webcam and drowsiness score
# --------------------------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError(
        "Could not open the webcam. Check camera permissions "
        "or try another camera index."
    )

closed_score = 0
DROWSY_THRESHOLD = 15

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Unable to read a frame from the webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        # Default display state
        status = "Face not detected"

        if len(faces) > 0:
            # Process the largest detected face
            x, y, w, h = max(
                faces,
                key=lambda face: face[2] * face[3]
            )

            cv2.rectangle(
                frame, (x, y), (x + w, y + h), (255, 200, 0), 2
            )

            # Detect eyes inside the face region
            face_gray = gray[y:y + h, x:x + w]
            face_color = frame[y:y + h, x:x + w]

            eyes = eye_cascade.detectMultiScale(
                face_gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(15, 15)
            )

            # Select up to two detected eye regions
            eyes = sorted(eyes, key=lambda eye: eye[0])[:2]
            eye_predictions = []

            for ex, ey, ew, eh in eyes:
                eye_image = face_color[ey:ey + eh, ex:ex + ew]

                if eye_image.size == 0:
                    continue

                # Match the CNN's training input: 80x80, 3 channels
                eye_image = cv2.resize(eye_image, (80, 80))
                eye_image = eye_image.astype("float32") / 255.0
                eye_input = np.expand_dims(eye_image, axis=0)

                # The sigmoid output represents class 1: closed eyes
                probability_closed = float(
                    model.predict(eye_input, verbose=0)[0][0]
                )

                is_closed = probability_closed >= 0.5
                eye_predictions.append(is_closed)

                # Drawing the eye detection box
                color = (0, 0, 255) if is_closed else (0, 255, 0)
                cv2.rectangle(
                    frame,
                    (x + ex, y + ey),
                    (x + ex + ew, y + ey + eh),
                    color,
                    2
                )

            # Increase the score only when every detected eye
            # is classified as closed.
            if eye_predictions:
                if all(eye_predictions):
                    closed_score += 1
                else:
                    closed_score = max(0, closed_score - 1)

                status = (
                    "DROWSINESS WARNING"
                    if closed_score >= DROWSY_THRESHOLD
                    else "Eyes closed" if all(eye_predictions)
                    else "Eyes open"
                )
            else:
                # No eye detections: do not treat this as a
                # confirmed closed-eye observation.
                closed_score = max(0, closed_score - 1)
                status = "Eyes not detected"

        else:
            # Avoid carrying a high score indefinitely when
            # no face is visible.
            closed_score = max(0, closed_score - 1)

        # --------------------------------------------------
        # 5. Display status and manage the alarm
        # --------------------------------------------------

        cv2.putText(
            frame,
            status,
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255) if closed_score >= DROWSY_THRESHOLD
            else (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Closed-eye score: {closed_score}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        if closed_score >= DROWSY_THRESHOLD and alarm is not None:
            if alarm_channel is None or not alarm_channel.get_busy():
                alarm_channel = alarm.play()

        elif alarm_channel is not None and alarm_channel.get_busy():
            alarm_channel.stop()
            alarm_channel = None

        cv2.imshow("Driver Drowsiness Detection", frame)

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()

    if alarm_channel is not None:
        alarm_channel.stop()

    if pygame.mixer.get_init():
        pygame.mixer.quit()
