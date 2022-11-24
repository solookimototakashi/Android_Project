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
    color = get_color_from_hex('#ffffff')

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
        self.title = '手書き数字認識テスト'

        self.image_width = 14   # ここを変更
        self.image_height = 14  # ここを変更
        self.color_setting = 3  # ここを変更。利用する学習済みモデルのカラー形式と同じにする

        # 学習を行う
        # self.model = learning.learn_MNIST()
        self.model = learning.load_MNIST(self.image_width,self.color_setting)
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
        # self.painter.export_to_png('canvas.png',float=0.1)  # 画像を一旦保存する       

        #2 各種設定  
        train_data_path = os.path.join(os.path.dirname(__file__),'Japanese_text_dataset') # ここを変更。
        for current_dir, sub_dirs, files_list in os.walk(train_data_path):
            if current_dir == train_data_path:
                fol_list = [os.path.join(current_dir,sub_dirs_item) for sub_dirs_item in sub_dirs]


        if self.color_setting == 1:
            img = cv2.imread("canvas.png", 0)
            # h, w = img.shape
            # img2 = img[0 : int(h*0.85), 0 : w]
            # cv2.imwrite("canvas.png", img2)
            # img = cv2.imread("canvas.png", 0)            
        elif self.color_setting == 3:
            img = cv2.imread("canvas.png", 1)
            # h, w, c = img.shape
            # img2 = img[0 : int(h*0.85), 0 : w]
            # cv2.imwrite("canvas.png", img2)               
            # img = cv2.resize(img, (image_width, image_height))
            plt.imshow(img)
        if self.color_setting == 1:
            plt.gray()  
            plt.show()
        elif self.color_setting == 3:
            plt.show()

        img = img.reshape(self.image_width, self.image_height, self.color_setting).astype('float32')/255 
        re_img = np.array([img])

        #5 予測と結果の表示等

        prediction = self.model.predict(re_img)
        result = prediction[0]

        for i, accuracy in enumerate(result):
            print('画像認識AIは「', os.path.basename(fol_list[i]), '」の確率を', int(accuracy * 100), '% と予測しました。')
            print('予測結果は、「', os.path.basename(fol_list[result.argmax()]),'」です。')
            print('-------------------------------------------------------')
            # print(' \n\n　＊　「確率精度が低い画像」や、「間違えた画像」を再学習させて、オリジナルのモデルを作成してみてください。')
            # print(' \n　＊　「間違えた画像」を数枚データセットに入れるだけで正解できる可能性が向上するようでした。')
            # print(' \n　＊　何度も実行すると「WARNING:tensorflow」が表示されますが、結果は正常に出力できるようでした。')

if __name__ == '__main__':
    Window.clearcolor = get_color_from_hex('#000000')   # ウィンドウの色を黒色に変更する
    MyPaintApp().run()