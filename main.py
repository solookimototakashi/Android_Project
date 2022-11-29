# coding=[utf8]

from random import random
from kivy.app import App
from kivy.config import Config

# 起動時の解像度の設定
Config.set('graphics', 'width', '1024')
Config.set('graphics', 'height', '768')  # 16:9
Config.set('graphics', 'resizable', False)  # ウインドウリサイズ禁止

from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.graphics import Color, Ellipse, Line
from kivy.properties import ObjectProperty
from kivy.factory import Factory
from kivy.uix.popup import Popup
from kivy.uix.behaviors import ToggleButtonBehavior
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.boxlayout import BoxLayout
from kivy.utils import get_color_from_hex   # 色の16進数表示を可能にする
from kivy.core.window import Window
import japanize_kivy

from kivy.lang import Builder

Builder.load_file('./window1.kv')
Builder.load_file('./window2.kv')

class MainRoot(BoxLayout):
    window1 = None
    window2 = None

    def __init__(self, **kwargs):
        # 起動時に各画面を作成して使い回す
        self.window1 = Factory.Window1()
        self.window2 = Factory.Window2()
        super(MainRoot, self).__init__(**kwargs)

    def change_disp(self):
        self.clear_widgets()
        self.add_widget(self.window1)

    def change_disp2(self):
        self.clear_widgets()
        self.add_widget(self.window2)

class MainApp(App): 
    def __init__(self, **kwargs):
        super(MainApp, self).__init__(**kwargs)
        self.title = '画面切り替えテスト'
    pass

# MyPaintWidget######################################################################################
class MyPaintWidget(Widget):
    #pass
    last_color = '' # 画面クリアを押された場合の最後の色
    line_width = 3  # 線の太さ

    def on_touch_down(self, touch):
        if Widget.on_touch_down(self, touch):
            return


        color = (random(), 1, 1)
        with self.canvas:
            #Color(*color, mode='hsv')
            #d = 30.
            #Ellipse(pos=(touch.x - d / 2, touch.y - d / 2), size=(d, d))
            touch.ud['line'] = Line(points=(touch.x, touch.y), width=self.line_width)

    def set_line_width(self, line_width=3):
        self.line_width = line_width

    def on_touch_move(self, touch):
        if touch.ud:    # スライダーを動かす際のエラーを解除するため
            print(touch.ud['line'].points)
            touch.ud['line'].points += [touch.x, touch.y]

    def set_color(self, new_color):
        ''' 塗る色を変更する '''
        self.last_color = new_color
        self.canvas.add(Color(*new_color))


class MyCanvasWidget(Widget):

    def clear_canvas(self):
        MyPaintWidget.clear_canvas(self)


class MyPaintApp(App):
    #paint_id = ObjectProperty(None)
    #self.painter.test # これでClearボタンにアクセス可能

    def __init__(self, **kwargs):
        super(MyPaintApp, self).__init__(**kwargs)
        self.title = '画像表示'
        self.loadfile = ObjectProperty(None)

    def build(self):
        parent = Widget()
        self.painter = MyCanvasWidget()

        # 起動時の色の設定を行う
        self.painter.ids['paint_area'].set_color(
            get_color_from_hex('#000000'))  #黒色を設定

        return self.painter

    def clear_canvas(self):
        '''
        画面をきれいにする。行うことは以下の2点
        1:画面をクリアーにする
        2:最後にセットしていた色をセットしなおす
        ''' 
        self.painter.ids['paint_area'].canvas.clear()
        self.painter.ids['paint_area'].set_color(self.painter.ids['paint_area'].last_color)

    def save_canvas(self):
        # 時間があるときに一時的にcanvas.beforeに背景を塗り潰す処理を加えるの
        # https://kivy.org/docs/api-kivy.core.window.html?highlight=screenshot#kivy.core.window.WindowBase.screenshot
        Window.screenshot();    # スクリーンショットを保存する
        #self.painter.export_to_png('a.png')    # 画像を保存する　ただしこのやり方だとウィンドウカラーが適用されないので描いていない部分が透明になる

    def load_img(self):
        Window.clearcolor = get_color_from_hex('#ffffff')   # ウィンドウの色を白色に変更する
        popup = Popup(title='PopupTest',
        content=Label(text='Hello world'),
        size_hint=(None, None), size=(400, 400))
        popup.open()

class ColorButton(ToggleButton):
    def _do_press(self):
        '''
        何も押されていない状態で設定が解除されるのを防ぐためToggleButtonの関数を継承して変更する 
        Source code for kivy.uix.behaviors.button
        https://kivy.org/docs/_modules/kivy/uix/behaviors/button.html
        '''

        if self.state == 'normal':
            # ボタンを押されてない場合は状態を変更する
            ToggleButtonBehavior._do_press(self)

####################################################################################################

class FileChooser_Root(BoxLayout):
    pass

class FileChooser(App):
    def build(self):
        return FileChooser_Root()

if __name__ == '__main__':
    Window.clearcolor = get_color_from_hex('#ffffff')   # ウィンドウの色を白色に変更する
    MainApp().run()
    # MyPaintApp().run()