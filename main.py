from segmentation import segmentation
from pose import get_input_point
from calculation import tarik_garis, elips
from calculation import load_and_preprocess

# 0 nose 1 shoulder 3elbow 5wrist 7hips 9knee 11ankle kanan semua kembalian get_input_point

TF_ENABLE_ONEDNN_OPTS=0

def main(path):

    kepala_x, kepala_y, lengan_x, lengan_y, paha_x, paha_y, koin_x = segmentation(path)

    path_get_point = load_and_preprocess(path)[1]
    coords, right_foot, left_foot = get_input_point(path_get_point)
    coords = coords.astype(int)
    #print(coords[0][1])

    lebar_koin = abs(max(koin_x)-min(koin_x))
    lebar_kepala = tarik_garis(kepala_y, coords[0][1], kepala_x)
    panjang_kepala = lebar_kepala*1.2
    lebar_lengan = tarik_garis(lengan_y, coords[3][1], lengan_x)
    lebar_paha = tarik_garis(paha_y, coords[9][1], paha_x)
    lebar_dada = abs(coords[1][0] - coords[2][0]) 
    lebar_perut = abs(coords[7][0] - coords[8][0]) 

    koefisien_koin = 2.7/lebar_koin

    lingkar_kepala = elips(lebar_kepala, lebar_kepala*1.2)*koefisien_koin
    lingkar_lengan = elips(lebar_lengan, lebar_lengan*0.87)*koefisien_koin
    lingkar_paha = elips(lebar_paha, lebar_paha*0.9)*koefisien_koin
    lingkar_dada = elips(lebar_dada, lebar_dada*0.8)*koefisien_koin
    lingkar_perut = elips(lebar_perut, lebar_perut*2)*koefisien_koin

    #kurang panjang badan, lingkar dada, lingkar perut
    shoulder2hips = abs(coords[1][1] - coords[7][1]) 
    panjang_badan = (panjang_kepala + shoulder2hips + (right_foot+left_foot)/2)*koefisien_koin

    return lingkar_kepala, lingkar_dada, lingkar_lengan, lingkar_perut, lingkar_paha, panjang_badan

print(main('images/baby2.jpg'))
