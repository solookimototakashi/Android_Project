from kivy.app import App
from kivy.uix.widget import Widget
from kivy.properties import StringProperty
from kivy.lang import Builder

from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.properties import BooleanProperty
from kivy.properties import NumericProperty
from kivy.clock import Clock

import os

# 日本語を適用させる
import japanize_kivy

# kv_dir = os.path.join(os.path.dirname(__file__), "template")
# kv_url = os.path.join(kv_dir, "test.kv")
kv_url = os.path.join(os.path.dirname(__file__), "test.kv")

# sample.kvを読み込む
Builder.load_file(kv_url)


class TextWidget(Widget):  # Kvファイル内でレイアウトしているクラス定義
    pass


class TestApp(App):  # メイン処理のクラス定義
    def __init__(self, **kwargs):
        super(TestApp, self).__init__(**kwargs)
        self.title = "Sample"  # ウィンドウのタイトル名

    def build(self):
        return TextWidget()


if __name__ == "__main__":
    TestApp().run()
