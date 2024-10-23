import base64
import os
from typing import Union

import cv2
import mediapipe as mp
import numpy as np
from fastapi import APIRouter
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from ..common.constants import MODEL_PATH
from ..schemas.detected_points import FullDetectedPoints, SideDetectedPoints
from ..schemas.image import Image
from ..services.posture_points import PosturePoints

router = APIRouter()
posture_points_inst = PosturePoints()

@router.post("/posture-points")
def posture_points(data: Image) -> SideDetectedPoints | FullDetectedPoints:
    base_options = python.BaseOptions(model_asset_path=os.path.abspath(os.curdir) + MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True
    )
    detector = vision.PoseLandmarker.create_from_options(options) 

    # STEP 3: Load the input image.
    # Decode the base64 string to a NumPy array
    decoded = base64.b64decode(data.base64)
    np_data = np.frombuffer(decoded,np.uint8)
    bgr_img = cv2.imdecode(np_data, flags=cv2.IMREAD_UNCHANGED)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=bgr_img)
    
    # STEP 4: Detect pose landmarks from the input image.
    detection_result = detector.detect(image)

    # STEP 5: Process the detection result. Visualize and draw circles.
    annotated_image, res = posture_points_inst.draw_landmarks_on_image(image.numpy_view(), detection_result)
    return res