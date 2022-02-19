import pandas as pd
import pickle

from IPython.display import Audio, display
from ipywidgets import Image, HTML, Layout, Output, VBox, HBox
from parameters import Parameters
from path import DATA
from pathlib import Path
from plotly import graph_objs as go


# Dataframe
file = Path('dataframe.json')
dataframe = Parameters(file)


# hover_details are those data that will
# be provided in the info table when hovering
# over the plot. Add any columns of df that you like.
HOVER_COLUMN = [
    dataframe.label_column,
    dataframe.call_identifier_column
]

distinct_colors_20 = [
    '#e6194b',
    '#3cb44b',
    '#ffe119',
    '#4363d8',
    '#f58231',
    '#911eb4',
    '#46f0f0',
    '#f032e6',
    '#bcf60c',
    '#fabebe',
    '#008080',
    '#e6beff',
    '#9a6324',
    '#fffac8',
    '#800000',
    '#aaffc3',
    '#808000',
    '#ffd8b1',
    '#000075',
    '#808080',
    '#ffffff',
    '#000000'
]

# Load dataframe
DF_NAME = DATA.joinpath('df_umap.pkl')
df = pd.read_pickle(DF_NAME)

# Load image data (deserialize)
with open(DATA.joinpath('image_data.pkl'), 'rb') as handle:
    image_data = pickle.load(handle)

if dataframe.call_identifier_column not in df.columns:
    print("Missing identifier column: ", dataframe.call_identifier_column)
    raise

labeltypes = sorted(
    list(
        set(df[dataframe.label_column])
    )
)

if len(labeltypes) <= len(distinct_colors_20):
    color_dict = dict(
        zip(
            labeltypes,
            distinct_colors_20[0:len(labeltypes)]
        )
    )
else:
    # if > 20 different labels, some will have the same color
    distinct_colors = distinct_colors_20 * len(labeltypes)
    color_dict = dict(
        zip(
            labeltypes,
            distinct_colors[0:len(labeltypes)]
        )
    )

# hover_details are those data that will be provided in the info table when hovering
# over datapoints

hover_details = HOVER_COLUMN

# Everything here is separated by labeltype, so that all datapoints from one specific label have their own trace

audio_dict = {} # dictionary that contains audio data for each labeltype
sr_dict = {} # dictionary that contains samplerate data for each labeltype
sub_df_dict = {} # dictionary that contains the dataframe for each labeltype

# build dictionary
for i, labeltype in enumerate(labeltypes):
    sub_df = df.loc[df.label == labeltype, :]
    sub_df_dict[i] = sub_df
    audio_dict[i] = sub_df[dataframe.audio_column]
    sr_dict[i] = sub_df['samplerate_hz']

# build traces
traces = []

for i, labeltype in enumerate(labeltypes):
    sub_df = sub_df_dict[i]
    trace = go.Scatter3d(
            x=sub_df.UMAP1,
            y=sub_df.UMAP2,
            z=sub_df.UMAP3,
            mode='markers',
            marker=dict(
                size=4,
                color=color_dict[labeltype],
                opacity=0.8
            ),
            name=labeltype,
            hovertemplate=[
                x for x in sub_df[dataframe.label_column]
            ]
    )
    traces.append(trace)

layout = go.Layout(
    scene=go.layout.Scene(
            xaxis=go.layout.scene.XAxis(title='UMAP1'),
            yaxis=go.layout.scene.YAxis(title='UMAP2'),
            zaxis=go.layout.scene.ZAxis(title='UMAP3')
    ),
    height=1000,
    width=1000
)

figure = go.Figure(data=traces, layout=layout)

fig = go.FigureWidget(figure)

# Initialize with any image (taking the first in the dictionary)

image_widget = Image(
    value=image_data[list(image_data.keys())[0]],
    layout=Layout(height='189px', width='300px')
)

details = HTML(
    layout=Layout(width='20%')
)


# define what happens when hovering over datapoint
def hover_fn(trace, points, state):
    if points.point_inds:
        # get the index of the data point being hovered
        trace_ind = points.trace_index
        sub_df = sub_df_dict[trace_ind]
        ind = points.point_inds[0]
        # Update image widget
        img_ind = sub_df.iloc[ind][dataframe.call_identifier_column]
        image_widget.value = image_data[img_ind]

        # Update details
        details.value = sub_df.iloc[ind][hover_details].to_frame().to_html()


for i in range(len(traces)):
    fig.data[i].on_hover(hover_fn)


# audio-playback function
def play_audio(ind, i):
    data = audio_dict[i]
    srs = sr_dict[i]
    display(
        Audio(
            data.iloc[ind],
            rate=srs.iloc[ind],
            autoplay=True
        )
    )

audio_widget = Output()
audio_widget.layout.visibility = 'hidden'


# define what happens when clicking on datapoint
def click_fn(trace, points, selector):
    if points.point_inds:

        # get the index of the data point being hovered
        trace_ind = points.trace_index
        ind = points.point_inds[0]

        # play audio
        with audio_widget:
            play_audio(ind, trace_ind)


for i in range(len(traces)):
    fig.data[i].on_click(click_fn)

# Put everything together.

# This renders the plot within jupyter notebook. Install voila to convert the notebook into a standalone web app
# (see https://voila.readthedocs.io/en/stable/using.html for details)
# Once installed, navigate to the jupyter notebook file in your file system and run
# > voila <path-to-02_viz_tool.ipynb>

# adjust vertical (VBox) and horizontal (HBoxes) if readibility is not good.

VBox([
    details,
    HBox(
        [image_widget, fig],
        layout=Layout(flex="none")
    ),
    audio_widget
])
