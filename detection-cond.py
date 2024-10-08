import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Load your image
image = cv2.imread('img/ex4.jpg')
is_side_view = False

def calculate_distance(p1, p2):
    """Calculates Euclidean distance between two points."""
    return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5

# Process the image
with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
    results = pose.process(image)

print(results.pose_landmarks)
# Determine side view
if results.pose_landmarks:
    shoulder_left = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    shoulder_right = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

    distance = calculate_distance(shoulder_left, shoulder_right)

    # Determine side view based on distance threshold
    threshold = 0.1  # Adjust threshold as needed
    if distance <= threshold:
        is_side_view = True
    else:
        is_side_view = False

# Find specific body points based on side view
if is_side_view:
    ear = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR]
    shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
    ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
     # Draw circles on the detected points
    cv2.circle(image, (int(ear.x * image.shape[1])+10, int(ear.y * image.shape[0])), 10, (0, 255, 0), -1)
    cv2.circle(image, (int(shoulder.x * image.shape[1]), int(shoulder.y * image.shape[0])), 10, (0, 255, 0), -1)
    cv2.circle(image, (int(hip.x * image.shape[1]), int(hip.y * image.shape[0])), 10, (0, 255, 0), -1)
    cv2.circle(image, (int(knee.x * image.shape[1]), int(knee.y * image.shape[0])), 10, (0, 255, 0), -1)
    cv2.circle(image, (int(ankle.x * image.shape[1]), int(ankle.y * image.shape[0])), 10, (0, 255, 0), -1)

    # Print or visualize the body points
    print("Side view detected:")
    print("Ear:", ear)
    print("Shoulder:", shoulder)
    print("Hip:", hip)
    print("Knee:", knee)
    print("Ankle:", ankle)
else:
    shoulder_left = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    shoulder_right = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    hip_left = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    hip_right = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
    knee_left = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
    knee_right = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]

     # Draw circles on the detected points
    cv2.circle(image, (int(shoulder_left.x * image.shape[1]), int(shoulder_left.y * image.shape[0])), 10, (0, 255, 0), -1)
    cv2.circle(image, (int(shoulder_right.x * image.shape[1]), int(shoulder_right.y * image.shape[0])), 10, (0, 255, 0), -1)
    cv2.circle(image, (int(knee_left.x * image.shape[1]), int(knee_left.y * image.shape[0])), 10, (0, 255, 0), -1)
    cv2.circle(image, (int(knee_right.x * image.shape[1]), int(knee_right.y * image.shape[0])), 10, (0, 255, 0), -1)
    cv2.circle(image, (int(hip_left.x * image.shape[1]), int(hip_left.y * image.shape[0])), 10, (0, 255, 0), -1)
    cv2.circle(image, (int(hip_right.x * image.shape[1]), int(hip_right.y * image.shape[0])), 10, (0, 255, 0), -1)

    # Print or visualize the body points
    print("Not side view:")
    print("Left shoulder:", shoulder_left)
    print("Right shoulder:", shoulder_right)
    print("Left hip:", hip_left)
    print("Right hip:", hip_right)
    print("Left knee:", knee_left)
    print("Right knee:", knee_right)

# Draw landmarks on the image (optional)
mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

# Display the image
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

if is_side_view:
    cv2.line(image, (int(ear.x * image.shape[1]-10), int(ear.y * image.shape[0])), (int(shoulder.x * image.shape[1]), int(shoulder.y * image.shape[0])), (0, 255, 0), 2)
    cv2.line(image, (int(shoulder.x * image.shape[1]), int(shoulder.y * image.shape[0])), (int(hip.x * image.shape[1]), int(hip.y * image.shape[0])), (0, 255, 0), 2)
    cv2.line(image, (int(hip.x * image.shape[1]), int(hip.y * image.shape[0])), (int(knee.x * image.shape[1]), int(knee.y * image.shape[0])), (0, 255, 0), 2)
    cv2.line(image, (int(knee.x * image.shape[1]), int(knee.y * image.shape[0])), (int(ankle.x * image.shape[1]), int(ankle.y * image.shape[0])), (0, 255, 0), 2)
else:
    cv2.line(image, (int(shoulder_left.x * image.shape[1]), int(shoulder_left.y * image.shape[0])), (int(shoulder_right.x * image.shape[1]), int(shoulder_right.y * image.shape[0])), (0, 255, 0), 2)
    cv2.line(image, (int(knee_left.x * image.shape[1]), int(knee_left.y * image.shape[0])), (int(knee_right.x * image.shape[1]), int(knee_right.y * image.shape[0])), (0, 255, 0), 2)
    cv2.line(image, (int(hip_left.x * image.shape[1]), int(hip_left.y * image.shape[0])), (int(hip_right.x * image.shape[1]), int(hip_right.y * image.shape[0])), (0, 255, 0), 2)

# Display the image with lines
cv2.imshow("Image with Key Posture Points and Lines", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
