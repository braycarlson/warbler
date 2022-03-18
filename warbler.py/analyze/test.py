import fitz

from path import ANALYZE
from spectrogram.axes import SpectrogramAxes
from spectrogram.plot import create_luscinia_spectrogram


image1 = ANALYZE.joinpath('luscinia/DbWY2017/STE01_DbWY2017.jpg')
image2 = ANALYZE.joinpath('luscinia/DbWY2017/STE02_DbWY2017.jpg')
image3 = ANALYZE.joinpath('luscinia/DbWY2017/STE03_DbWY2017.jpg')

# Image box
width = 321.900390625
height = 98.97236633300781
offset = 40
expand = 325

document = fitz.open()

toc = []

current = document.new_page()

# Set page size
current.set_mediabox(
    fitz.Rect(
        0.0,
        0.0,
        700,
        750
    )
)

# Title
current.insert_textbox(
    fitz.Rect(
        0.0,
        offset,
        700,
        100
    ),
    'Spectrogram',
    fontsize=24,
    align=1
)

# Image 1 text
current.insert_textbox(
    fitz.Rect(
        10,
        230,
        680,
        750
    ) * current.rotation_matrix,
    'Luscinia',
    fontsize=10,
    align=0,
    rotate=270
)

current.insert_image(
    fitz.Rect(
        0,
        offset,
        width + expand,
        expand + height + offset
    ),
    filename=image1,
    keep_proportion=True
)

# Image 2 text
current.insert_textbox(
    fitz.Rect(
        15,
        395,
        680,
        750
    ) * current.rotation_matrix,
    'Python: Filtered',
    fontsize=10,
    align=0,
    rotate=270
)

current.insert_image(
    fitz.Rect(
        0,
        height + (offset * 2),
        width + expand,
        (expand + height * 2) + (offset * 4)
    ),
    filename=image2,
    keep_proportion=True
)

# Image 3 text
current.insert_textbox(
    fitz.Rect(
        15,
        565,
        680,
        750
    ) * current.rotation_matrix,
    'Python: Segmented',
    fontsize=10,
    align=0,
    rotate=270
)

current.insert_image(
    fitz.Rect(
        0,
        (height * 2) + (offset * 4),
        width + expand,
        (expand + height * 3) + (offset * 6)
    ),
    filename=image3,
    keep_proportion=True
)

path = ANALYZE.joinpath('test.pdf')
document.save(path)
