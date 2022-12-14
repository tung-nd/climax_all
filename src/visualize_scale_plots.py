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

markers = ['s', '^', 'x', 'o', 'p']


def plot(ax, data, title, show_legend=True):
    count = 0
    for idx, key in enumerate(data["5.625"]):
        ax.plot(data["5.625"][key]["x"], data["5.625"][key]["y"], color=f"C{idx}", label=f"CliMax{key}-5.625", marker=markers[idx])
        count += 1

    for idx, key in enumerate(data["1.40625"]):
        ax.plot(data["1.40625"][key]["x"], data["1.40625"][key]["y"], color=f"C{idx+count}", label=f"CliMax{key}-1.40625", marker=markers[idx])
        count += 1

    if "ifs" in data.keys():
        ax.plot(range(1, 6), np.ones(5)* data["ifs"], color=f"C{count+1}", label="IFS", linestyle='--', alpha=0.5)
    if "fourcastnet" in data.keys():
        ax.plot(range(1, 6), np.ones(5)* data["fourcastnet"], color=f"C{count+2}", label="FourCastNet-0.25", linestyle='--', alpha=0.5)
    if "pangu" in data.keys():
        ax.plot(range(1, 6), np.ones(5)* data["pangu"], color=f"C{count+3}", label="PanguWeather-0.25", linestyle='--', alpha=0.5)

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
@click.option('--output', '-o', default="scale_plots.png")
def main(jsons, output):

    fig, axs = plt.subplots(1, len(jsons), figsize=(6*len(jsons), 4), squeeze=True)
    for idx, jsonf in enumerate(jsons):
        with open(jsonf, 'r') as f:
            data = json.load(f)
        ax = axs[idx] if type(axs) is np.ndarray else axs
        plot(ax, data, title=data["task"])
    
    fig.savefig(output, bbox_inches="tight", dpi=300)
    
if __name__ == "__main__":
    main()
