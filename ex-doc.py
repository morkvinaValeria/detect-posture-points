import cv2
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision 
from enum import Enum

import numpy as np
import os

model_path = '/pose_landmarker_heavy.task'
mp_pose = mp.solutions.pose

class Side(Enum):
  LEFT = "left"
  RIGHT = "right"

def calculate_distance(p1, p2):
  """Calculates Euclidean distance between two points."""
  return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5

def check_view(left_landmark, right_landmark, distance_threshold=0.1):
  """Determine side view based on distance threshold"""
  distance = calculate_distance(left_landmark, right_landmark)
  is_side_view = True if distance <= distance_threshold else False
  print("is_side_view result:", is_side_view)

  return is_side_view

def check_side(nose, left_shoulder):
  return Side.LEFT if nose.x < left_shoulder.x else Side.RIGHT

def draw_landmarks_on_image(rgb_image, detection_result, circle_radius=7, circle_color=(0, 0, 255)):
  """Draws pose landmarks on the image, including circles on shoulders.

  Args:
      rgb_image: The input image in RGB format.
      detection_result: The pose detection result from MediaPipe.
      circle_radius: The radius of the circles to draw on the shoulders (default: 7).
      circle_color: The color of the circles (default: blue, BGR format).

  Returns:
      The annotated image with landmarks and circles.
  """

  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)): 

    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
        annotated_image,
        pose_landmarks_proto,
        solutions.pose.POSE_CONNECTIONS,
        solutions.drawing_styles.get_default_pose_landmarks_style()) 


    # Get shoulder landmarks and draw circles
    left_shoulder = pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    nose = pose_landmarks[mp_pose.PoseLandmark.NOSE]

    is_side_view = check_view(left_shoulder, right_shoulder)
    if is_side_view:
      print(check_side(nose, left_shoulder))

    # Convert normalized coordinates to image coordinates (assuming 0-255 pixel range)
    image_width, image_height = annotated_image.shape[1], annotated_image.shape[0]
    # Draw circles on shoulders using BGR color format
    cv2.circle(annotated_image, (int(left_shoulder.x * image_width), int(left_shoulder.y * image_height)), circle_radius, circle_color, -1)
    cv2.circle(annotated_image, (int(right_shoulder.x * image_width), int(right_shoulder.y * image_height)), circle_radius, circle_color, -1)

  return annotated_image

# STEP 2: Create an PoseLandmarker object.
base_options = python.BaseOptions(model_asset_path=os.path.dirname(__file__) + model_path)
options = vision.PoseLandmarkerOptions(
  base_options=base_options,
  output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options) 


# STEP 3: Load the input image.
image = mp.Image.create_from_file("img/ex8.jpg") 


# STEP 4: Detect pose landmarks from the input image.
detection_result = detector.detect(image)

# STEP 5: Process the detection result. Visualize and draw circles.
annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
cv2.imshow("", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)) 

cv2.waitKey(0)
cv2.destroyAllWindows()