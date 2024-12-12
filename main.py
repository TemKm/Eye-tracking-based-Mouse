import cv2
import mediapipe as mp
import pyautogui

camera = cv2.VideoCapture(0)
face_mesh_detector = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_width, screen_height = pyautogui.size()

while True:
    ret, video_frame = camera.read()
    if not ret:
        break

    video_frame = cv2.flip(video_frame, 1)
    rgb_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
    face_mesh_results = face_mesh_detector.process(rgb_frame)
    facial_landmarks = face_mesh_results.multi_face_landmarks
    frame_height, frame_width, _ = video_frame.shape

    adjusted_x_min = int(frame_width * 0.1)
    adjusted_x_max = int(frame_width * 0.9)
    adjusted_y_min = int(frame_height * 0.1)
    adjusted_y_max = int(frame_height * 0.9)

    if facial_landmarks:
        landmarks = facial_landmarks[0].landmark
        for idx, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_width)
            y = int(landmark.y * frame_height)

            if adjusted_x_min <= x <= adjusted_x_max and adjusted_y_min <= y <= adjusted_y_max:
                cv2.circle(video_frame, (x, y), 3, (0, 255, 0))
                if idx == 1:
                    normalized_x = (landmark.x - 0.1) / 0.8
                    normalized_y = (landmark.y - 0.1) / 0.8
                    screen_x = screen_width * min(max(normalized_x, 0), 1)
                    screen_y = screen_height * min(max(normalized_y, 0), 1)
                    pyautogui.moveTo(screen_x, screen_y)

        right_eye_landmarks = [landmarks[374], landmarks[386]]
        for landmark in right_eye_landmarks:
            x = int(landmark.x * frame_width)
            y = int(landmark.y * frame_height)
            if adjusted_x_min <= x <= adjusted_x_max and adjusted_y_min <= y <= adjusted_y_max:
                cv2.circle(video_frame, (x, y), 3, (0, 255, 255))
        if (right_eye_landmarks[0].y - right_eye_landmarks[1].y) < 0.004:
            pyautogui.click()
            pyautogui.sleep(1)

    cv2.imshow('Eye Controlled Mouse', video_frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

camera.release()
cv2.destroyAllWindows()
