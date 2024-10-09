import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision 
from enum import Enum

import numpy as np
import os

model_path = '/pose_landmarker_heavy.task'
mp_pose = mp.solutions.pose

class Side(Enum):
  LEFT = "LEFT"
  RIGHT = "RIGHT"

def calculate_distance(p1, p2):
  """Calculates Euclidean distance between two points."""
  return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5

def is_side_view(left_landmark, right_landmark, distance_threshold=0.1):
  distance = calculate_distance(left_landmark, right_landmark)
  is_side_view = True if distance <= distance_threshold else False

  return is_side_view

def check_side(nose, left_shoulder):
  return Side.LEFT if nose.x < left_shoulder.x else Side.RIGHT

def draw_side_view_circles(image, side, pose_landmarks, image_width, image_height, circle_radius=8, circle_color=(128, 0, 255), thickness=-1):
  ear_threshold = 15 if side == Side.LEFT else -15 
  
  ear = pose_landmarks[mp_pose.PoseLandmark[side.value + "_EAR"]]
  shoulder = pose_landmarks[mp_pose.PoseLandmark[side.value + "_SHOULDER"]]
  hip = pose_landmarks[mp_pose.PoseLandmark[side.value + "_HIP"]]
  knee = pose_landmarks[mp_pose.PoseLandmark[side.value + "_KNEE"]]
  ankle = pose_landmarks[mp_pose.PoseLandmark[side.value + "_ANKLE"]]
  for landmark in [ear, shoulder, hip, knee, ankle]:
    cv2.circle(image, (int(landmark.x * image_width) + ear_threshold, int(landmark.y * image_height)), circle_radius, circle_color, thickness)

def draw_full_view_circles(image, pose_landmarks, image_width, image_height, circle_radius=8, circle_color=(128, 0, 255), thickness=-1):
  for side in list(Side):
    shoulder = pose_landmarks[mp_pose.PoseLandmark[side.value + "_SHOULDER"]]
    hip = pose_landmarks[mp_pose.PoseLandmark[side.value + "_HIP"]]
    knee = pose_landmarks[mp_pose.PoseLandmark[side.value + "_KNEE"]]
    for landmark in [shoulder, hip, knee]:
      cv2.circle(image, (int(landmark.x * image_width), int(landmark.y * image_height)), circle_radius, circle_color, thickness)
      
def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  if len(pose_landmarks_list) != 1:
    raise Exception("Image should contain only 1 pose")
  
  annotated_image = np.copy(rgb_image)
  image_width, image_height = annotated_image.shape[1], annotated_image.shape[0]
  pose_landmarks = pose_landmarks_list[0]
  ## Draw the pose landmarks.
  # pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
  # pose_landmarks_proto.landmark.extend([
  #     landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
  # ])
  # solutions.drawing_utils.draw_landmarks(
  #     annotated_image,
  #     pose_landmarks_proto,
  #     solutions.pose.POSE_CONNECTIONS,
  #     solutions.drawing_styles.get_default_pose_landmarks_style()) 

  left_shoulder = pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
  right_shoulder = pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
  nose = pose_landmarks[mp_pose.PoseLandmark.NOSE]

  if is_side_view(left_shoulder, right_shoulder):
    side = check_side(nose, left_shoulder)
    draw_side_view_circles(annotated_image, side, pose_landmarks, image_width, image_height)
  else:
    draw_full_view_circles(annotated_image, pose_landmarks, image_width, image_height)

  return annotated_image

# STEP 2: Create an PoseLandmarker object.
base_options = python.BaseOptions(model_asset_path=os.path.dirname(__file__) + model_path)
options = vision.PoseLandmarkerOptions(
  base_options=base_options,
  output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options) 


# STEP 3: Load the input image.
image = mp.Image.create_from_file("img/ex3.jpg") 


# STEP 4: Detect pose landmarks from the input image.
detection_result = detector.detect(image)

# STEP 5: Process the detection result. Visualize and draw circles.
annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
cv2.imshow("", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)) 

cv2.waitKey(0)
cv2.destroyAllWindows()