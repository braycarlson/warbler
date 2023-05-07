from datatype.dataset import Dataset
from datatype.imaging import draw_segment


def main():
    dataset = Dataset('segment')
    dataframe = dataset.load()

    mask = dataframe['folder'] == 'YYLb_STE2017'
    subset = dataframe.loc[mask].reset_index(drop=True)

    by = ['duration']
    ascending = [True]

    subset.sort_values(
        ascending=ascending,
        by=by,
        inplace=True
    )

    spectrogram = subset.spectrogram.to_numpy()

    fig, ax = draw_segment(spectrogram)

    fig.savefig(
        'grid.png',
        bbox_inches='tight',
        dpi=72,
        format='png',
        pad_inches=0
    )


if __name__ == '__main__':
    main()
