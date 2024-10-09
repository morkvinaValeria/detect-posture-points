import os

import mediapipe as mp
from fastapi import APIRouter
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from ..common.constants import MODEL_PATH
from ..services.posture_points import PosturePoints

router = APIRouter()
posture_points_inst = PosturePoints()

@router.get("/posture-points")
def posture_points():
    base_options = python.BaseOptions(model_asset_path=os.path.abspath(os.curdir) + MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True
    )
    detector = vision.PoseLandmarker.create_from_options(options) 

    # STEP 3: Load the input image.
    image = mp.Image.create_from_file(os.path.abspath(os.curdir) +"\img\ex1.jpg") 

    # STEP 4: Detect pose landmarks from the input image.
    detection_result = detector.detect(image)

    # STEP 5: Process the detection result. Visualize and draw circles.
    annotated_image, res = posture_points_inst.draw_landmarks_on_image(image.numpy_view(), detection_result)
    return res