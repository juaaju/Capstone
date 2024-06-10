from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp
import numpy as np
from calculation import calculate_distance

#Fungsi Ngambil Koordinat Pose
def get_input_point(image):
  base_options = python.BaseOptions(model_asset_path='model/pose_landmarker_lite.task')
  options = vision.PoseLandmarkerOptions(
      base_options=base_options,
      output_segmentation_masks=True)
  detector = vision.PoseLandmarker.create_from_options(options)
  image = mp.Image.create_from_file(image)

  detection_result = detector.detect(image)
  pose_landmarks = detection_result.pose_landmarks

  nose = pose_landmarks[0][0]
  right_shoulder = pose_landmarks[0][12]
  right_elbow = pose_landmarks[0][14]
  right_wrist = pose_landmarks[0][16]
  left_shoulder = pose_landmarks[0][11]
  left_elbow = pose_landmarks[0][13]
  left_wrist = pose_landmarks[0][15]
  left_hip = pose_landmarks[0][23]
  left_knee = pose_landmarks[0][25]
  left_ankle = pose_landmarks[0][27]
  right_hip = pose_landmarks[0][24]
  right_knee = pose_landmarks[0][26]
  right_ankle = pose_landmarks[0][28]

  right_hand = calculate_distance(right_shoulder.x*image.width, right_shoulder.y*image.height, right_elbow.x*image.width, right_elbow.y*image.height) + calculate_distance(right_elbow.x*image.width, right_elbow.y*image.height, right_wrist.x*image.width, right_wrist.y*image.height)

  left_hand = calculate_distance(left_shoulder.x*image.width, left_shoulder.y*image.height, left_elbow.x*image.width, left_elbow.y*image.height) + calculate_distance(left_elbow.x*image.width, left_elbow.y*image.height, left_wrist.x*image.width, left_wrist.y*image.height)

  right_foot = calculate_distance(right_hip.x*image.width, right_hip.y*image.height, right_knee.x*image.width, right_knee.y*image.height) + calculate_distance(right_knee.x*image.width, right_knee.y*image.height, right_ankle.x*image.width, right_ankle.y*image.height)

  left_foot = calculate_distance(left_hip.x*image.width, left_hip.y*image.height, left_knee.x*image.width, left_knee.y*image.height) + calculate_distance(left_knee.x*image.width, left_knee.y*image.height, left_ankle.x*image.width, left_ankle.y*image.height)

  coords = np.array([
      [nose.x*image.width,nose.y*image.height],
      [right_shoulder.x*image.width,right_shoulder.y*image.height],
      [left_shoulder.x*image.width,left_shoulder.y*image.height],
      [right_elbow.x*image.width,right_elbow.y*image.height],
      [left_elbow.x*image.width,left_elbow.y*image.height],
      [right_wrist.x*image.width,right_wrist.y*image.height],
      [left_wrist.x*image.width,left_wrist.y*image.height],
      [right_hip.x*image.width,right_hip.y*image.height],
      [left_hip.x*image.width,left_hip.y*image.height],
      [right_knee.x*image.width,right_knee.y*image.height],
      [left_knee.x*image.width,left_knee.y*image.height],
      [right_ankle.x*image.width,right_ankle.y*image.height],
      [left_ankle.x*image.width,left_ankle.y*image.height],
      ])
  
  # 0 nose 1 shoulder 3elbow 5wrist 7hips 9knee 11ankle kanan semua

  return coords, right_foot, left_foot