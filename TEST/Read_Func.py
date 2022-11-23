#1 ライブラリのインポート等

from keras.models import load_model
import os
import cv2 
import matplotlib.pyplot as plt
import numpy as np


#2 各種設定
def Read_Func(folder,recognise_image):
    # recognise_image = '日.png' # ここを変更
                            # 画像認識したい画像ファイル名。（実行前に認識したい画像ファイルを1つアップロードしてください）
                            # エラーが出る場合は、「パスをコピー」の利用や、「ファイル名の変更」でファイル名をコピー&ペーストしてみてください
                            # 文字形式の兼ね合いでエラーが出る時もあるようでした
                            # 直接「プ」を入力するとエラーがでました。右端の「」内の文字をコピー&ペーストで成功しました：「プ.png」

    image_width = 14   # ここを変更
                    # 利用する学習済みモデルの横の幅のピクセル数と同じにする
    image_height = 14  # ここを変更
                    # 利用する学習済みモデルの縦の高さのピクセル数と同じにする
    color_setting = 1  # ここを変更。利用する学習済みモデルのカラー形式と同じにする
                    # 「1」はモノクロ・グレースケール。「3」はカラー。


    #3 各種読み込み

    model = load_model('model.h5')  # ここを変更
                                    # 読み込む学習済みモデルを入れます

    # モノクロ・グレー形式の学習済みモデルを読み込む例：color_setting = 1 の学習済みモデルを使う場合  
    #model = load_model('keras_cnn_japanese_handwritten_gray14*14_model.h5')  

    # カラー形式の学習済みモデルを読み込む例：color_setting = 3 の学習済みモデルを使う場合  
    #model = load_model('keras_cnn_japanese_handwritten_color14*14_model.h5') 


    #4 画像の表示・各種設定等

    if color_setting == 1:
        img = cv2.imread(recognise_image, 0)   
    elif color_setting == 3:
        img = cv2.imread(recognise_image, 1)
        img = cv2.resize(img, (image_width, image_height))
        plt.imshow(img)
    if color_setting == 1:
        plt.gray()  
        plt.show()
    elif color_setting == 3:
        plt.show()


    img = img.reshape(image_width, image_height, color_setting).astype('float32')/255 


    #5 予測と結果の表示等

    prediction = model.predict(np.array([img]))
    result = prediction[0]

    for i, accuracy in enumerate(result):
        print('画像認識AIは「', os.path.basename(folder[i]), '」の確率を', int(accuracy * 100), '% と予測しました。')

        print('-------------------------------------------------------')
        print('予測結果は、「', os.path.basename(folder[result.argmax()]),'」です。')
        print(' \n\n　＊　「確率精度が低い画像」や、「間違えた画像」を再学習させて、オリジナルのモデルを作成してみてください。')
        print(' \n　＊　「間違えた画像」を数枚データセットに入れるだけで正解できる可能性が向上するようでした。')
        print(' \n　＊　何度も実行すると「WARNING:tensorflow」が表示されますが、結果は正常に出力できるようでした。')