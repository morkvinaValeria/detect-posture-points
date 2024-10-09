import mediapipe as mp
import numpy as np

from ..common.enums import Side
from ..utils.calculate_distance import calculate_distance


class PosturePoints:
    def __init__(self):
        self._mp_pose = mp.solutions.pose

    def is_side_view(self, left_landmark, right_landmark, distance_threshold=0.1):
        distance = calculate_distance(left_landmark, right_landmark)
        is_side_view = True if distance <= distance_threshold else False

        return is_side_view

    def check_side(self, nose, left_shoulder):
        return Side.LEFT if nose.x < left_shoulder.x else Side.RIGHT

    def draw_side_view_circles(self, image, side, pose_landmarks, image_width, image_height, circle_radius=8, circle_color=(128, 0, 255), thickness=-1):
        # ear_threshold = 15 if side == Side.LEFT else -15  
        ear = pose_landmarks[self._mp_pose.PoseLandmark[side.value + "_EAR"]]
        shoulder = pose_landmarks[self._mp_pose.PoseLandmark[side.value + "_SHOULDER"]]
        hip = pose_landmarks[self._mp_pose.PoseLandmark[side.value + "_HIP"]]
        knee = pose_landmarks[self._mp_pose.PoseLandmark[side.value + "_KNEE"]]
        ankle = pose_landmarks[self._mp_pose.PoseLandmark[side.value + "_ANKLE"]]
        # for landmark in [ear, shoulder, hip, knee, ankle]:
        #     cv2.circle(image, (int(landmark.x * image_width) + ear_threshold, int(landmark.y * image_height)), circle_radius, circle_color, thickness)
        return {
            "landmarks": {"ear": ear, "shoulder": shoulder, "hip": hip, "knee": knee, "ankle": ankle},
            "sideView": True,
            "side": side.value
        }

    def draw_full_view_circles(self, image, pose_landmarks, image_width, image_height, circle_radius=8, circle_color=(128, 0, 255), thickness=-1):
        res = {
            "landmarks": {},
            "sideView": False
        }
        for side in list(Side):
            shoulder = pose_landmarks[self._mp_pose.PoseLandmark[side.value + "_SHOULDER"]]
            hip = pose_landmarks[self._mp_pose.PoseLandmark[side.value + "_HIP"]]
            knee = pose_landmarks[self._mp_pose.PoseLandmark[side.value + "_KNEE"]]
            for landmark in [shoulder, hip, knee]:
                side_val = side.value.lower()
                if landmark == shoulder:
                    res['landmarks'][side_val + "Shoulder"] = shoulder
                elif landmark == hip:
                    res['landmarks'][side_val + "Hip"] = hip
                else:
                    res['landmarks'][side_val + "Knee"] = knee

                # cv2.circle(image, (int(landmark.x * image_width), int(landmark.y * image_height)), circle_radius, circle_color, thickness)
        return res
    
    def draw_landmarks_on_image(self, rgb_image, detection_result):
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

        left_shoulder = pose_landmarks[self._mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks[self._mp_pose.PoseLandmark.RIGHT_SHOULDER]
        nose = pose_landmarks[self._mp_pose.PoseLandmark.NOSE]

        if self.is_side_view(left_shoulder, right_shoulder):
            side = self.check_side(nose, left_shoulder)
            res = self.draw_side_view_circles(annotated_image, side, pose_landmarks, image_width, image_height)
        else:
            res = self.draw_full_view_circles(annotated_image, pose_landmarks, image_width, image_height)

        return annotated_image, res
