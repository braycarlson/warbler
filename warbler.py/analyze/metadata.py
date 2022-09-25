from constant import PICKLE
from datatype.file import File


def main():
    pickle = [
        file
        for file in PICKLE.glob('*.xz')
    ]

    for file in pickle:
        filename = file.name
        file = File(filename)

        file = file.load()
        print(file)


if __name__ == '__main__':
    main()
