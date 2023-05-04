from IPython.display import display
from ipywidgets import (
    Button,
    HBox,
    Layout,
    Output
)


class Gallery:
    def __init__(self, collection):
        self.collection = collection

        self.index = 0
        self.length = len(self.collection)
        self.output = Output()
        self.next = Button(description='Next')
        self.previous = Button(description='Previous')

    def create(self):
        with self.output:
            self.output.clear_output(True)

            widget = self.collection[self.index]
            plot = widget.create()

            button = HBox(
                [self.previous, self.next],
                layout=Layout(
                    align_items='center',
                    display='flex',
                    flex_flow='row',
                    align_content='stretch',
                    justify_content='center'
                )
            )

            button.add_class('button')

            self.set_event_listener()

            display(plot)
            display(button)

        return self.output

    def on_next(self, button):
        if self.index == self.length - 1:
            self.index = 0
        else:
            self.index = self.index + 1

        self.create()

    def on_previous(self, button):
        if self.index == 0:
            self.index = self.length - 1
        else:
            self.index = self.index - 1

        self.create()

    def set_event_listener(self):
        self.next.on_click(self.on_next)
        self.previous.on_click(self.on_previous)
