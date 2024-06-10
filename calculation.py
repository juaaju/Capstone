import cv2
import numpy as np
import math
from skimage.io import imread
from skimage.transform import resize

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def load_and_preprocess(path):
  X_test = np.zeros((1, 512, 384, 3), dtype=np.uint8)
  sizes_test = []

  img = imread(path)[:,:,:3]
  sizes_test.append([img.shape[0], img.shape[1]])
  img = resize(img, (512, 384), mode='constant', preserve_range=True)
  X_test[0] = img
  save_path = 'images/baby.jpg'
  cv2.imwrite(save_path, img)
  return X_test, save_path

def get_coordinates(segmentation_result):
  segmentation_coords = np.argwhere(segmentation_result[0] > 0.95)
  x_coordinates = []
  y_coordinates = []
  for coords in segmentation_coords:
    x_coordinates.append(coords[1])
    y_coordinates.append(coords[0])

  return x_coordinates, y_coordinates

# perhitungan keliling elips
def elips(sb_mayor, sb_minor):
  return 0.5*3.14*(sb_mayor + sb_minor)

def tarik_garis(sb_y, pose_coords_y, sb_x):
    idx_y = []
    for idx, y in enumerate(sb_y):
        if pose_coords_y == y:
            idx_y.append(idx)
    
    x_val = []
    for i in idx_y:
        x_val.append(sb_x[i])

    lebar = abs(max(x_val) - min(x_val))

    #print(x_val)

    return lebar