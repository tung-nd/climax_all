import json
from email.policy import default

import click
import matplotlib.pyplot as plt
import numpy as np

try:
    import matplotx
    plt.style.use([matplotx.styles.tab10, matplotx.styles.dufte])
except ImportError:
    matplotx = None
    pass

markers = ['s', '^', 'x', 'o', 'p']


def plot(ax, data, title, show_legend=True):
    count = 0
    for idx, key in enumerate(data):
        if key != "task":
            ax.plot(data[key]["x"], data[key]["y"], color=f"C{idx}", label=f"CliMax{key}", marker=markers[idx])
            count += 1

    ax.set_xlim(0.9, 5.1)
    ax.set_xticks(range(1, 6))
    ax.ticklabel_format(style="plain")
    ax.set_title(title)
    ax.set_xlabel("Pretraining datasize")
    if matplotx is not None:
        matplotx.ylabel_top("RMSE", ax)
        if show_legend:
            matplotx.line_labels()
    else:
        ax.set_ylabel("RMSE")
        if show_legend:
            ax.legend()

@click.command()
@click.argument('jsons', nargs=-1)
@click.option('--variable', default="Z500")
@click.option('--output', '-o', default="scale_plots.png")
def main(jsons, variable, output):

    fig, axs = plt.subplots(1, len(jsons), figsize=(6*len(jsons), 4), squeeze=True)
    for idx, jsonf in enumerate(jsons):
        with open(jsonf, 'r') as f:
            data = json.load(f)
        ax = axs[idx] if type(axs) is np.ndarray else axs
        plot(ax, data[variable], title=data[variable]["task"])
    
    fig.savefig(output, bbox_inches="tight", dpi=300)
    
if __name__ == "__main__":
    main()
