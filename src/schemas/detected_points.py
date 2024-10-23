from pydantic import BaseModel

from ..common.enums import Side

class Point(BaseModel):
    x: float
    y: float
    z: float

class FullLandmarks(BaseModel):
    rightShoulder: Point
    leftShoulder: Point
    rightHip: Point
    leftHip: Point
    rightKnee: Point
    leftKnee: Point

class SideLandmarks(BaseModel):
    ear: Point
    shoulder: Point
    hip: Point
    knee: Point
    ankle: Point

class FullDetectedPoints(BaseModel):
    sideView: bool = False
    landmarks: FullLandmarks
   

class SideDetectedPoints(BaseModel):
    sideView: bool = True
    landmarks: SideLandmarks
    side: Side
 