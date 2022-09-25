from IPython.display import display
from ipywidgets import (
    Button,
    GridspecLayout,
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
            widget.create()

            grid = GridspecLayout(2, 2)
            grid[0, :] = widget.get_plot()
            grid[1, 0] = self.previous
            grid[1, 1] = self.next

            self.set_event_listener()
            display(grid)

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
