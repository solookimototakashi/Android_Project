from kivy.app import App
from kivy.uix.widget import Widget
from kivy.properties import StringProperty
from kivy.core.window import Window
from kivy.lang import Builder

# 日本語を適用させる
import japanize_kivy

# sample.kvを読み込む
Builder.load_file("C://Users/もちねこ/Desktop/MyVSProject/Android_Project/Sample.kv")

# Windowの背景色を変更 (黒 → 白)
Window.clearcolor = (1, 1, 1, 1)

# ウィジェットクラス
class Sample(Widget):
    # プロパティの追加
    text = StringProperty()

    def __init__(self, **kwargs):
        super(Sample, self).__init__(**kwargs)
        # 初期値の設定
        self.text = "kivyの練習です！"

    # コールバック用の関数を定義
    def callback(self, **kwargs):
        # 呼び出された際に、textの値を更新
        self.text = "kivy : 基礎学習"


# メインクラス
class Test(App):
    def build(self):
        # ウィジェットクラスを返す
        return Sample()


if __name__ == "__main__":
    # アプリを起動
    Test().run()
