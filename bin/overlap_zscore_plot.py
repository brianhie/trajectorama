import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys

if __name__ == '__main__':
    data = []
    with open(sys.argv[1]) as f:
        for line in f:
            data.append(line.rstrip().split())

    df = pd.DataFrame(
        data,
        columns=[ 'Modality', 'Network', 'NumOverlap', 'PctOverlap',
                  'ZScore', 'PermPVal'],
    )
    df = df.astype({
        'NumOverlap': int,
        'PctOverlap': float,
        'ZScore': float,
        'PermPVal': float,
    })

    plt.figure()
    sns.barplot(data=df, x='Modality', y='ZScore', hue='Network',
                hue_order=[ 'target', 'baseline' ])
    plt.ylim([ -5, 48 ])
    plt.savefig('overlap_zscore.svg')
