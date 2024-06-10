import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from calculation import load_and_preprocess, get_coordinates

#print("versi tf: ", tf.__version__)

def segmentation(path_image):
    model_kepala = load_model('model/model_head.h5', safe_mode=False)
    model_lengan = load_model('model/model_upper_arm.h5', safe_mode=False)
    model_paha = load_model('model/model_thigh.h5', safe_mode=False)
    model_koin = load_model('model/model_coin.h5', safe_mode=False)

    input_image = load_and_preprocess(path_image)[0]

    prediksi_kepala = model_kepala.predict(input_image)
    prediksi_lengan = model_lengan.predict(input_image)
    prediksi_paha = model_paha.predict(input_image)
    prediksi_koin = model_koin.predict(input_image)

    segmentasi_kepala = (prediksi_kepala > 0.5).astype(np.uint8)
    segmentasi_lengan = (prediksi_lengan > 0.5).astype(np.uint8)
    segmentasi_paha = (prediksi_paha > 0.5).astype(np.uint8)
    segmentasi_koin = (prediksi_koin > 0.5).astype(np.uint8)

    # koordinat
    kepala_x, kepala_y = get_coordinates(segmentasi_kepala)
    lengan_x, lengan_y = get_coordinates(segmentasi_lengan)
    paha_x, paha_y = get_coordinates(segmentasi_paha)
    koin_x, koin_y = get_coordinates(segmentasi_koin)

    #print("kepala_y",kepala_y)

    return kepala_x, kepala_y, lengan_x, lengan_y, paha_x, paha_y, koin_x

