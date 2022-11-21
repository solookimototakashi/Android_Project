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

kv_dir = os.path.join(os.path.dirname(__file__), "template")
kv_url = os.path.join(kv_dir, "test.kv")
# sample.kvを読み込む
Builder.load_file(kv_url)


class TextWidget(Widget):
    text = StringProperty("")


#     def __init__(self, **kwargs):
#         super(TextWidget, self).__init__(**kwargs)
#         self.text = "default"

#     def onButtonClick(self):
#         self.text = "Hello Kivy"


class TestApp(App):
    def __init__(self, **kwargs):
        super(TestApp, self).__init__(**kwargs)
        self.title = "KivyTest"

    def build(self):
        return TextWidget()


if __name__ == "__main__":
    TestApp().run()
