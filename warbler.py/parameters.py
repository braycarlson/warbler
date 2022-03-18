import json

from types import SimpleNamespace


class Parameters(SimpleNamespace):
    def __init__(self, path):
        self.path = path
        self.file = open(self.path, 'r')
        self.data = json.load(self.file)

        super().__init__(**self.data)

    def _reload(self):
        self.close()

        self.file = open(self.path, 'r')
        self.data = json.load(self.file)
        super().__init__(**self.data)

    def update(self, key, value):
        if hasattr(self, key):
            setattr(self, key, value)

        if key in self.data:
            self.data[key] = value

    def save(self):
        self.close()

        with open(self.path, 'w+') as file:
            json.dump(self.data, file, indent=4)

        self._reload()

    def close(self):
        self.file.close()


# class Parameters(SimpleNamespace):
#     def __init__(self, file):
#         self.file = file
#         super().__init__(**self.file)

#     def __reduce__(self):
#         return (
#             self.__class__,
#             (self.file, )
#         )

#     def to_json(self):
#         return json.dumps(
#             self,
#             default=lambda x: x.__dict__,
#             sort_keys=False,
#             indent=4
#         )
