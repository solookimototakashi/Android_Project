from kivy.app import App
from kivy.uix.widget import Widget
from kivy.properties import StringProperty
from kivy.uix.screenmanager import ScreenManager, Screen
import japanize_kivy
# from kivy.lang import Builder
# import os

# kv_dir = os.path.join(os.path.dirname(__file__),"template")
# kv_file_path = os.path.join(kv_dir,'test.kv')
# Builder.load_file(kv_file_path)

class MainScreen(Screen):
    pass


class SubScreen(Screen):
    pass


class ScreenApp(App):
    def build(self):
        self.sm = ScreenManager()
        self.sm.add_widget(MainScreen(name='main'))
        self.sm.add_widget(SubScreen(name='sub'))
        return self.sm


if __name__ == '__main__':
    ScreenApp().run()