import json

import click
import matplotlib.pyplot as plt
import numpy as np

try:
    import matplotx
    plt.style.use([matplotx.styles.tab10, matplotx.styles.dufte])
except ImportError:
    matplotx = None
    pass



def plot(ax, data, title, show_legend=True):
    xs = data.keys()
    xs = [x for x in xs if x != 'task']
    ys = [data[x] for x in xs]
    colors = [f"C{idx}" for idx in range(len(ys))]

    y_min = min(*ys)
    bottom = 0.9 * y_min
    ys = [y - bottom for y in ys]

    ax.bar(xs, ys, width=0.5, bottom=bottom, color=colors)

    ax.xaxis.set_tick_params(labelsize='small')

    ax.set_title(title)
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
@click.option('--output', '-o', default="local_plots.png")
def main(jsons, output):

    fig, axs = plt.subplots(1, len(jsons), figsize=(8*len(jsons), 4), squeeze=True)
    for idx, jsonf in enumerate(jsons):
        with open(jsonf, 'r') as f:
            data = json.load(f)
        ax = axs[idx] if type(axs) is np.ndarray else axs
        plot(ax, data, title=data["task"])
    
    fig.savefig(output, bbox_inches="tight", dpi=300)
    
if __name__ == "__main__":
    main()
