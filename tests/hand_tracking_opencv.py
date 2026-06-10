import cv2
import mediapipe as mp


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]


def count_fingers(hand_landmarks, handedness_label):
    """
    Count how many fingers are likely extended.

    This is a simple rule-based method:
    - For index, middle, ring, pinky:
      fingertip above PIP joint means extended.
    - For thumb:
      horizontal direction is used, depending on left/right hand.
    """
    landmarks = hand_landmarks.landmark
    fingers = []

    # Thumb
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]

    if handedness_label == "Right":
        thumb_is_open = thumb_tip.x < thumb_ip.x
    else:
        thumb_is_open = thumb_tip.x > thumb_ip.x

    fingers.append(thumb_is_open)

    # Other four fingers
    for tip_id, pip_id in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
        tip = landmarks[tip_id]
        pip = landmarks[pip_id]
        finger_is_open = tip.y < pip.y
        fingers.append(finger_is_open)

    return sum(fingers), fingers


def get_gesture_name(finger_count, fingers):
    """
    Return a simple gesture name based on open fingers.

    fingers order:
    [thumb, index, middle, ring, pinky]
    """
    thumb, index, middle, ring, pinky = fingers

    if finger_count == 0:
        return "Fist"
    if finger_count == 5:
        return "Open Palm"
    if index and middle and not ring and not pinky:
        return "Peace"
    if index and not middle and not ring and not pinky:
        return "Pointing"
    if thumb and not index and not middle and not ring and not pinky:
        return "Thumb"
    return f"{finger_count} fingers"


def draw_hand_box(image, hand_landmarks):
    """
    Draw a bounding rectangle around the detected hand.
    """
    h, w, _ = image.shape

    xs = [landmark.x for landmark in hand_landmarks.landmark]
    ys = [landmark.y for landmark in hand_landmarks.landmark]

    x_min = int(min(xs) * w)
    x_max = int(max(xs) * w)
    y_min = int(min(ys) * h)
    y_max = int(max(ys) * h)

    padding = 20
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w, x_max + padding)
    y_max = min(h, y_max + padding)

    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    return x_min, y_min, x_max, y_max


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot open camera.")
        print("Try changing camera index from 0 to 1 in cv2.VideoCapture(0).")
        return

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    ) as hands:
        while True:
            success, frame = cap.read()

            if not success:
                print("Error: Failed to read frame from camera.")
                break

            # Mirror image for more natural webcam interaction
            frame = cv2.flip(frame, 1)

            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = hands.process(rgb_frame)
            rgb_frame.flags.writeable = True

            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks,
                    results.multi_handedness,
                ):
                    handedness_label = handedness.classification[0].label

                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )

                    x_min, y_min, _, _ = draw_hand_box(frame, hand_landmarks)

                    finger_count, fingers = count_fingers(
                        hand_landmarks,
                        handedness_label,
                    )
                    gesture_name = get_gesture_name(finger_count, fingers)

                    label = f"{handedness_label}: {gesture_name}"

                    cv2.putText(
                        frame,
                        label,
                        (x_min, max(30, y_min - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

            cv2.putText(
                frame,
                "Press Q to quit",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("OpenCV Hand Tracking", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
