import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import math

def get_joint_angle(p1, p2, p3):
    """
    Returns the angle (in degrees) at point p2
    formed by points p1 - p2 - p3
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)

    v1 = p1 - p2
    v2 = p3 - p2

    dot = np.dot(v1, v2)
    mag = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6

    angle = math.acos(np.clip(dot / mag, -1.0, 1.0))
    return math.degrees(angle)


def smooth_values(buffer, window=5):
    if len(buffer) < window:
        return np.mean(buffer)
    return np.mean(list(buffer)[-window:])


mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose_model = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


video_path = "input_video.mp4"
cap = cv2.VideoCapture(video_path)

angle_buffer = {
    "left_elbow": deque(maxlen=10),
    "right_elbow": deque(maxlen=10),
    "spine": deque(maxlen=10)
}

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose_model.process(rgb_frame)
    frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

    warnings = []

    if result.pose_landmarks:
        landmarks = result.pose_landmarks.landmark

        l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
        l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y]
        l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y]

        r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
        r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
        r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]

        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x,
               landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x,
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]

        left_elbow_angle = get_joint_angle(l_shoulder, l_elbow, l_wrist)
        right_elbow_angle = get_joint_angle(r_shoulder, r_elbow, r_wrist)
        spine_angle = get_joint_angle(l_shoulder, hip, knee)

        angle_buffer["left_elbow"].append(left_elbow_angle)
        angle_buffer["right_elbow"].append(right_elbow_angle)
        angle_buffer["spine"].append(spine_angle)

        left_elbow_avg = smooth_values(angle_buffer["left_elbow"])
        right_elbow_avg = smooth_values(angle_buffer["right_elbow"])
        spine_avg = smooth_values(angle_buffer["spine"])

        if not 30 <= left_elbow_avg <= 160:
            warnings.append("❌ Left elbow angle out of range")

        if not 30 <= right_elbow_avg <= 160:
            warnings.append("❌ Right elbow angle out of range")

        if spine_avg < 160:
            warnings.append("❌ Back bending detected")

        if abs(left_elbow_avg - right_elbow_avg) > 15:
            warnings.append("❌ Arm movement not symmetric")
    
        mp_draw.draw_landmarks(
            frame,
            result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        text_y = 30
        for msg in warnings:
            cv2.putText(frame, msg, (10, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            text_y += 25

        if not warnings:
            cv2.putText(frame, "✅ Good Form",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 200, 0), 2)

    cv2.imshow("Exercise Form Checker", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
