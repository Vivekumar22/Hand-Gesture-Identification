import cv2
import mediapipe as mp


mp_hands = mp.solutions.hands
hands = mp_hands.Hands()


mp_drawing = mp.solutions.drawing_utils


def count_fingers(hand_landmarks, frame_width, frame_height):
    tip_ids = [4, 8, 12, 16, 20]
    finger_count = 0


    if hand_landmarks.landmark[tip_ids[0]].x * frame_width < hand_landmarks.landmark[tip_ids[1]].x * frame_width:
        finger_count += 1


    for id in range(1, 5):
        if hand_landmarks.landmark[tip_ids[id]].y * frame_height < hand_landmarks.landmark[
            tip_ids[id] - 2].y * frame_height:
            finger_count += 1


    if finger_count == 0:
        thumb_tip_y = hand_landmarks.landmark[4].y * frame_height
        pinky_tip_y = hand_landmarks.landmark[20].y * frame_height
        if thumb_tip_y > pinky_tip_y:
            finger_count = 5

    return finger_count



cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        continue


    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    results = hands.process(frame_rgb)


    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)


            frame_height, frame_width, _ = frame.shape


            finger_count = count_fingers(hand_landmarks, frame_width, frame_height)


            cv2.putText(frame, "Finger Count: " + str(finger_count), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                        2, cv2.LINE_AA)

    cv2.imshow('Hand Recognition', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



