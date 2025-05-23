import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Get standard angles from landmarks
def calculate_standard_angles(landmarks, w, h):
    def get_point(part):
        return [landmarks[part].x * w, landmarks[part].y * h]

    angles = {
        "left_knee": calculate_angle(get_point(mp_pose.PoseLandmark.LEFT_HIP),
                                      get_point(mp_pose.PoseLandmark.LEFT_KNEE),
                                      get_point(mp_pose.PoseLandmark.LEFT_ANKLE)),

        "right_knee": calculate_angle(get_point(mp_pose.PoseLandmark.RIGHT_HIP),
                                       get_point(mp_pose.PoseLandmark.RIGHT_KNEE),
                                       get_point(mp_pose.PoseLandmark.RIGHT_ANKLE)),

        "left_hip": calculate_angle(get_point(mp_pose.PoseLandmark.LEFT_SHOULDER),
                                     get_point(mp_pose.PoseLandmark.LEFT_HIP),
                                     get_point(mp_pose.PoseLandmark.LEFT_KNEE)),

        "right_hip": calculate_angle(get_point(mp_pose.PoseLandmark.RIGHT_SHOULDER),
                                      get_point(mp_pose.PoseLandmark.RIGHT_HIP),
                                      get_point(mp_pose.PoseLandmark.RIGHT_KNEE))
    }

    angles["landmarks"] = landmarks  # 👈 Correctly added for Warrior II logic
    return angles


# Custom pose logic
def get_pose_angles(standard_angles, pose_name):
    angles = standard_angles.copy()
    if pose_name == "Tree Pose":
        if angles["left_knee"] > angles["right_knee"]:
            angles["standing_leg_knee"] = angles["left_knee"]
            angles["bent_leg_knee"] = angles["right_knee"]
            angles["standing_leg_hip"] = angles["left_hip"]
        else:
            angles["standing_leg_knee"] = angles["right_knee"]
            angles["bent_leg_knee"] = angles["left_knee"]
            angles["standing_leg_hip"] = angles["right_hip"]
    elif pose_name == "Warrior II":
        # Determine front and back leg using x-position (camera facing user, so flipped)
        left_ankle_x = standard_angles["landmarks"][mp_pose.PoseLandmark.LEFT_ANKLE].x
        right_ankle_x = standard_angles["landmarks"][mp_pose.PoseLandmark.RIGHT_ANKLE].x
        
        if left_ankle_x < right_ankle_x:
            # Left leg is in front
            angles["front_knee"] = angles["left_knee"]
            angles["back_knee"] = angles["right_knee"]
        else:
            # Right leg is in front
            angles["front_knee"] = angles["right_knee"]
            angles["back_knee"] = angles["left_knee"]
    elif pose_name == "Child's Pose":
        angles["knees"] = (angles["left_knee"] + angles["right_knee"]) / 2
        angles["hips"] = (angles["left_hip"] + angles["right_hip"]) / 2
    return angles


# Draw angle labels
def draw_angles(image, angles, landmarks, pose_name):
    for joint, angle in angles.items():
        try:
            if "left" in joint:
                point = landmarks[getattr(mp_pose.PoseLandmark, joint.upper())]
            elif "right" in joint:
                point = landmarks[getattr(mp_pose.PoseLandmark, joint.upper())]
            else:
                continue

            cx, cy = int(point.x * image.shape[1]), int(point.y * image.shape[0])
            cv2.putText(image, f"{angle:.0f} deg", (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        except:
            continue

# Define angle targets
POSES = {
    "Tree Pose": {
        "standing_leg_knee": (170, 190),
        "bent_leg_knee": (45, 135),
        "standing_leg_hip": (170, 190)
    },
    "Warrior II": {
        "front_knee": (80, 100),
        "back_knee": (170, 190)
    },
    "Child's Pose": {
        "knees": (30, 60),
        "hips": (100, 140)
    }
}

# Ask user
pose_name = "Warrior II"  # Change this to test different poses

# Start camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera not working.")
    exit()

print("Starting in 3...")
time.sleep(1)
print("2...")
time.sleep(1)
print("1...")
time.sleep(1)
print("Start")

start = time.time()
duration = 60
correct_pose_time = 0
prev = start

while time.time() - start < duration:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        angles = calculate_standard_angles(lm, frame.shape[1], frame.shape[0])
        pose_angles = get_pose_angles(angles, pose_name)

        # Check angles
        all_good = True
        feedback = []
        for joint, (min_a, max_a) in POSES[pose_name].items():
            a = pose_angles.get(joint, 0)
            if not (min_a <= a <= max_a):
                all_good = False
                feedback.append(f"{joint}: {'+' if a < min_a else '-'}{abs(a - (min_a if a < min_a else max_a)):.1f} deg")

        now = time.time()
        if all_good:
            correct_pose_time += now - prev
            cv2.putText(frame, "POSE OK", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "POSE WRONG", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        prev = now

        # Draw feedback
        for i, text in enumerate(feedback[:3]):
            cv2.putText(frame, text, (10, 100 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Draw landmarks
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
        )

        draw_angles(frame, pose_angles, lm, pose_name)

    else:
        cv2.putText(frame, "No person detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show timer and results
    elapsed = time.time() - start
    remaining = max(0, duration - elapsed)
    cv2.putText(frame, f"Time Left: {remaining:.1f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Correct Time: {correct_pose_time:.1f}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display
    cv2.imshow("Yoga Pose Correction", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pose.close()

print(f"\nResults for {pose_name}:")
print(f"Time in correct pose: {correct_pose_time:.2f} seconds")
print(f"Accuracy: {(correct_pose_time / duration) * 100:.1f}%")