# --- Kivy 関連 --- #
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


class MyPaintWidget(Widget):
    line_width = 20  # 線の太さ
    color = get_color_from_hex('#ffffff')  # 線の色

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

    def build(self):
        self.painter = MyCanvasWidget()
        # 起動時の色の設定を行う
        self.painter.ids['paint_area'].set_color()
        return self.painter

    def clear_canvas(self):
        self.painter.ids['paint_area'].canvas.clear()
        self.painter.ids['paint_area'].set_color()  # クリアした後に色を再びセット

    def predict(self):
        pass  # ここで自分の書いた手書き数字の認識結果を表示させたい


if __name__ == '__main__':
    Window.clearcolor = get_color_from_hex('#000000')   # ウィンドウの色を黒色に変更する
    MyPaintApp().run()