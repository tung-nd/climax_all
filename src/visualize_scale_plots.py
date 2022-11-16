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

    ax.plot(range(1, 6), np.ones(5)* data["ifs"], color=f"C{count+1}", label="IFS", linestyle='--', alpha=0.5)
    ax.plot(range(1, 6), np.ones(5)* data["fourcastnet"], color=f"C{count+2}", label="FourCastNet-0.25", linestyle='--', alpha=0.5)
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

def main():
    # TODO: read a json file instead of hardcoding
    data = {}
    data["5.625"] = {}
    data["5.625"]["1024"] = {"x": [1, 2, 3, 4, 5], "y": [271, 252, 248, 247, 243]}
    data["5.625"]["512"] = {"x": [1, 5], "y": [289, 257]}
    data["5.625"]["256"] = {"x": [1, 5], "y": [300, 282]}
    data["5.625"]["128"] = {"x": [1, 5], "y": [350, 336]}    
    data["1.40625"] = {}
    data["1.40625"]["1024"] = {"x": [5], "y": [193]}
    data["ifs"] = 155
    data["fourcastnet"] = 220
    data["pangu"] = 134.5
    data["task"] = "Z500 (3 days)"
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), squeeze=True)
    plot(axs[0], data, title=data["task"])
    data = {}
    data["5.625"] = {}
    data["5.625"]["1024"] = {"x": [1, 2, 3, 4, 5], "y": [1.71, 1.64, 1.63, 1.62, 1.60]}
    data["5.625"]["512"] = {"x": [1, 5], "y": [1.76, 1.70]}
    data["5.625"]["256"] = {"x": [1, 5], "y": [1.88, 1.84]}
    data["5.625"]["128"] = {"x": [1, 5], "y": [2.10, 2.06]}
    data["1.40625"] = {}
    data["1.40625"]["1024"] = {"x": [5], "y": [1.42]}
    data["ifs"] = 1.37
    data["fourcastnet"] = 1.5
    data["pangu"] = 1.14
    data["task"] = "T850 (3 days)"
    plot(axs[1], data, title=data["task"])
    fig.savefig("scale_plots.png", bbox_inches="tight", dpi=300)
    
if __name__ == "__main__":
    main()