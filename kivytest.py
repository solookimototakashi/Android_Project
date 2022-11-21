from kivy.app import App
from kivy.uix.widget import Widget
from kivy.properties import StringProperty


class TextWidget(Widget):
    text = StringProperty("")

    def __init__(self, **kwargs):
        super(TextWidget, self).__init__(**kwargs)
        self.text = "No Text"

    def onButtonClick(self):
        self.text = self.ids.textInput.text


class TestApp(App):
    def __init__(self, **kwargs):
        super(TestApp, self).__init__(**kwargs)
        self.title = "KivyTest"


if __name__ == "__main__":
    TestApp().run()
