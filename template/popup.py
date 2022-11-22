from kivy.app import App
from kivy.uix.popup import Popup
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import ObjectProperty
import japanize_kivy

class PopupMenu(BoxLayout):
    popup_close = ObjectProperty(None)


class PopupTest(BoxLayout):
    def popup_open(self):
        content = PopupMenu(popup_close=self.popup_close)
        self.popup = Popup(title='Popup Test', content=content, size_hint=(0.5, 0.5), auto_dismiss=False)
        self.popup.open()

    def popup_close(self):
        self.popup.dismiss()


class PopupApp(App):
    def build(self):
        return PopupTest()


if __name__ == '__main__':
    PopupApp().run()