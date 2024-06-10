from flask import Flask, request, jsonify
from segmentation import segmentation
from pose import get_input_point
from calculation import tarik_garis, elips, load_and_preprocess
import os
import requests
from PIL import Image
from io import BytesIO

app = Flask(__name__)

def main(image_path):
    try:
        print("Starting segmentation...")
        kepala_x, kepala_y, lengan_x, lengan_y, paha_x, paha_y, koin_x = segmentation(image_path)
        print(f"Segmentation completed: {kepala_x}, {kepala_y}, {lengan_x}, {lengan_y}, {paha_x}, {paha_y}, {koin_x}")

        print("Starting load and preprocess...")
        path_get_point = load_and_preprocess(image_path)[1]
        print(f"Load and preprocess completed: {path_get_point}")

        print("Starting get input point...")
        coords, right_foot, left_foot = get_input_point(path_get_point)
        coords = coords.astype(int)
        print(f"Get input point completed: {coords}, {right_foot}, {left_foot}")

        lebar_koin = abs(max(koin_x) - min(koin_x))
        lebar_kepala = tarik_garis(kepala_y, coords[0][1], kepala_x)
        panjang_kepala = lebar_kepala * 1.2
        lebar_lengan = tarik_garis(lengan_y, coords[3][1], lengan_x)
        lebar_paha = tarik_garis(paha_y, coords[9][1], paha_x)
        lebar_dada = abs(coords[1][0] - coords[2][0])
        lebar_perut = abs(coords[7][0] - coords[8][0])

        koefisien_koin = 2.7 / lebar_koin

        lingkar_kepala = elips(lebar_kepala, lebar_kepala * 1.2) * koefisien_koin
        lingkar_lengan = elips(lebar_lengan, lebar_lengan * 0.87) * koefisien_koin
        lingkar_paha = elips(lebar_paha, lebar_paha * 0.9) * koefisien_koin
        lingkar_dada = elips(lebar_dada, lebar_dada * 0.8) * koefisien_koin
        lingkar_perut = elips(lebar_perut, lebar_perut * 2) * koefisien_koin

        shoulder2hips = abs(coords[1][1] - coords[7][1])
        panjang_badan = (panjang_kepala + shoulder2hips + (right_foot + left_foot) / 2) * koefisien_koin

        result = {
            'lingkar_kepala': lingkar_kepala,
            'lingkar_dada': lingkar_dada,
            'lingkar_lengan': lingkar_lengan,
            'lingkar_perut': lingkar_perut,
            'lingkar_paha': lingkar_paha,
            'panjang_badan': panjang_badan
        }

        print(f"Processing completed successfully: {result}")
        return result
    except Exception as e:
        error_message = f"Error in main processing: {str(e)}"
        print(error_message)
        raise ValueError(error_message)

@app.route('/predictions', methods=['POST'])
def create_prediction():
    try:
        data = request.json
        image_url = data['imageUrl']

        print(f"Fetching image from URL: {image_url}")
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        image_path = os.path.join('/tmp', 'temp_image.jpg')
        image.save(image_path)

        print("Processing image...")
        result = main(image_path)
        print("Image processed successfully.")

        return jsonify(result)
    except Exception as e:
        error_message = f"Error creating prediction: {str(e)}"
        print(error_message)
        return jsonify({'error': error_message}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
