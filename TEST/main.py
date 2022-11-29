# --- Kivy 関連 --- #
#1 各種インポート
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
from kivy.app import App
from kivy.config import Config

# Config関係は他のモジュールがインポートされる前に行う必要があるため、ここ記述
Config.set('graphics', 'width', '300')
Config.set('graphics', 'height', '340')
Config.write()

from kivy.uix.widget import Widget
from kivy.graphics import Color, Line
from kivy.utils import get_color_from_hex
from keras.models import load_model
from kivy.core.window import Window
# ----------------- #

from PIL import Image
import numpy as np
import learning
import cv2
import os
import matplotlib.pyplot as plt
from kivy.graphics import (Canvas, Translate, Fbo, ClearColor,
                           ClearBuffers, Scale)

import Read_Func


class MyPaintWidget(Widget):
    line_width = 5  # 線の太さ
    color = get_color_from_hex('#ffffff') #ffffff

    def on_touch_down(self, touch):
        if Widget.on_touch_down(self, touch):
            return

        with self.canvas:
            touch.ud['line'] = Line(points=(touch.x, touch.y), width=self.line_width)

    def on_touch_move(self, touch):
        touch.ud['line'].points += [touch.x, touch.y]

    def set_color(self):
        self.canvas.add(Color(*self.color))


class MyCanvasWidget(Widget):
    def clear_canvas(self):
        MyPaintWidget.clear_canvas(self)


class MyPaintApp(App):

    def __init__(self, **kwargs):
        super(MyPaintApp, self).__init__(**kwargs)
        train_data_path = os.path.join(os.path.dirname(__file__),'Japanese_text_dataset') # ここを変更。Colaboratoryにアップロードしたzipファイルを解凍後の、データセットのフォルダ名を入力
        self.title = '手書き数字認識テスト'

        self.image_width = 64   # ここを変更
        self.image_height = 64  # ここを変更
        self.color_setting = 1  # ここを変更。利用する学習済みモデルのカラー形式と同じにする

        for curDir, dirs, files in os.walk(train_data_path):
            if curDir == train_data_path:
                self.folders = [curDir + r"\\" + i for i in dirs]

        # 学習を行う
        # self.model = learning.learn_MNIST()
        # self.model = learning.load_MNIST(self.image_width,self.color_setting)
        #学習済モデルをロード
        self.model = load_model("cnn_model_weight.h5")
        self.model.summary()


    def build(self):
        self.painter = MyCanvasWidget()
        # 起動時の色の設定を行う
        self.painter.ids['paint_area'].set_color()
        return self.painter

    def clear_canvas(self):
        self.painter.ids['paint_area'].canvas.clear()
        self.painter.ids['paint_area'].set_color()  # クリアした後に色を再びセット

    def predict(self):
        self.painter.export_to_png('canvas.png',float=1)  # 画像を一旦保存する       
        if self.color_setting == 1:
            img = cv2.imread("canvas.png", cv2.IMREAD_GRAYSCALE )
            h, w = img.shape
            img2 = img[0 : int(h*0.85), 0 : w]
            img2 = cv2.resize(img2, (self.image_width, self.image_height))
            img2 = cv2.bitwise_not(img2) # キャンバスが黒なので色反転
            cv2.imwrite("canvas.png", img2)
            img = cv2.imread("canvas.png", cv2.IMREAD_GRAYSCALE )            
        elif self.color_setting == 3:
            img = cv2.imread("canvas.png", cv2.IMREAD_COLOR )
            h, w, c = img.shape
            img2 = img[0 : int(h*0.85), 0 : w, 0 : c]
            img2 = cv2.resize(img2, (self.image_width, self.image_height))
            img2 = cv2.bitwise_not(img2) # キャンバスが黒なので色反転
            cv2.imwrite("canvas.png", img2)
            img = cv2.imread("canvas.png", cv2.IMREAD_COLOR )               
            # img = cv2.resize(img, (self.image_width, self.image_height))
            plt.imshow(img)
        if self.color_setting == 1:
            # plt.gray()  
            plt.show()
        elif self.color_setting == 3:
            plt.show()
        if self.color_setting == 1:
            # img = img.reshape(self.image_width, self.image_height).astype('float32')/255 
            re_img = np.array([img])
        elif self.color_setting == 3:
            img = img.reshape(self.image_width, self.image_height, self.color_setting).astype('float32')/255 
            re_img = np.array([img])

        #5 予測と結果の表示等

        prediction = self.model.predict(re_img, batch_size=64, verbose=0, steps=None)
        result = prediction[0]

        for i, accuracy in enumerate(result):
            print('画像認識AIは「', os.path.basename(self.folders[i]), '」の確率を', int(accuracy * 100), '% と予測しました。')
            print('予測結果は、「', os.path.basename(self.folders[result.argmax()]),'」です。')
            print('-------------------------------------------------------')
            # print(' \n\n　＊　「確率精度が低い画像」や、「間違えた画像」を再学習させて、オリジナルのモデルを作成してみてください。')
            # print(' \n　＊　「間違えた画像」を数枚データセットに入れるだけで正解できる可能性が向上するようでした。')
            # print(' \n　＊　何度も実行すると「WARNING:tensorflow」が表示されますが、結果は正常に出力できるようでした。')

if __name__ == '__main__':
    Window.clearcolor = get_color_from_hex('#000000')   # ウィンドウの色を黒色に変更する
    MyPaintApp().run()