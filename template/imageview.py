from kivy.app import App
from kivy.config import Config
from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty
from kivy.lang import Builder
import japanize_kivy

#ウインドウの幅と高さの設定
Config.set('graphics', 'width', 700)
Config.set('graphics', 'height', 500)
#1でサイズ変更可、0はサイズ変更不可
Config.set('graphics', 'resizable', 1)
#kvファイルを指定
Builder.load_file("imageview.kv")

class MyLayout(Widget):
    pass
        
class DispimageApp(App):
    def build(self):
        self.title = "window"
        return MyLayout()

if __name__ == '__main__':
    DispimageApp().run()