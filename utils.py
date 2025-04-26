import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter1d

custom_colors = {
        "random": "#FFB6C1",
        "mixed": "#D2B48C",
        "nested": "#D8BFD8",
        "constituency": "#ADD8E6",
        "dependency": "#90EE90"
    }

def depth_distribution(data, output_name, title):
    plt.figure(figsize=(12, 6))
    colors = plt.get_cmap("tab10")

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12

    for idx, (label, data) in enumerate(data.items()):
        x = sorted(map(int, data.keys()))
        y = [data[str(k)] for k in x]

        y_total = sum(y)
        y = [v / y_total for v in y]

        x_new = np.linspace(min(x), max(x), 500)
        spl = make_interp_spline(x, y, k=3)
        y_smooth = spl(x_new)
        y_smoothed = gaussian_filter1d(y_smooth, sigma=3)
        
        plt.plot(x_new, y_smoothed, label=label)
        plt.fill_between(x_new, y_smoothed, alpha=0.2)

    # plt.title(title)
    plt.xlabel(title)
    plt.ylabel('Percentage')
    plt.xlim(0, 40) 
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4, frameon=False)
    plt.tight_layout()
    plt.savefig(output_name)

def leaves_distribution(data, output_name, title):
    plt.figure(figsize=(12, 6))
    colors = plt.get_cmap("tab10")

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12

    for idx, (label, data) in enumerate(data.items()):
        x = sorted(map(int, data.keys()))
        y = [data[str(k)] for k in x]

        y_total = sum(y)
        print(sum(y))
        y = [v / y_total for v in y]

        x_new = np.linspace(min(x), max(x), 500)
        spl = make_interp_spline(x, y, k=3)
        y_smooth = spl(x_new)
        y_smoothed = gaussian_filter1d(y_smooth, sigma=3)

        plt.plot(x_new, y_smoothed, label=label, color=colors(idx))
        plt.fill_between(x_new, y_smoothed, alpha=0.2)

    # plt.title(title)
    plt.xlabel(title)
    plt.ylabel('Percentage')
    plt.xlim(0, 200) 
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4, frameon=False)
    plt.tight_layout()
    plt.savefig(output_name)

def symmetry_distribution(data, output_name, title):
    plt.figure(figsize=(12, 6))
    colors = plt.get_cmap("tab10")

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12

    for idx, (label, data) in enumerate(data.items()):
        x = sorted(map(float, data.keys()))
        y = [data[str(k)] for k in x]

        y_total = sum(y)
        print(sum(y))
        y = [v / y_total for v in y]

        x_new = np.linspace(min(x), max(x), 500)
        spl = make_interp_spline(x, y, k=3)
        y_smooth = spl(x_new)
        y_smoothed = gaussian_filter1d(y_smooth, sigma=3)

        plt.plot(x_new, y_smoothed, label=label, color=colors(idx))
        plt.fill_between(x_new, y_smoothed, alpha=0.2)

    # plt.title(title)
    plt.xlabel(title)
    plt.ylabel('Percentage')
    # plt.xlim(0, 15) 
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4, frameon=False)
    plt.tight_layout()
    plt.savefig(output_name)

def branch_distribution(data, output_name, title):
    plt.figure(figsize=(12, 6))
    colors = plt.get_cmap("tab10")

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12

    for idx, (label, data) in enumerate(data.items()):
        x = sorted(map(float, data.keys()))
        y = [data[str(k)] for k in x]

        y_total = sum(y)
        print(sum(y))
        y = [v / y_total for v in y]

        x_new = np.linspace(min(x), max(x), 500)
        spl = make_interp_spline(x, y, k=3)
        y_smooth = spl(x_new)
        y_smoothed = gaussian_filter1d(y_smooth, sigma=3)
        y_smoothed = y_smoothed / np.sum(y_smoothed)

        plt.plot(x_new, y_smoothed, label=label, color=colors(idx))
        plt.fill_between(x_new, y_smoothed, alpha=0.2)

    # plt.title(title)
    plt.xlabel(title)
    plt.ylabel('Percentage')
    plt.xlim(0, 15) 
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4, frameon=False)
    plt.tight_layout()
    plt.savefig(output_name)

with open('features/nested.json', 'r') as file:
    nested = json.load(file)

with open('features/constituency.json', 'r') as file:
    constituency = json.load(file)

with open('features/mixed.json', 'r') as file:
    mixed = json.load(file)

with open('features/dependency.json', 'r') as file:
    dependency = json.load(file)

depth = {
    "nested": nested['depth'],
    "constituency": constituency['depth'],
    "mixed": mixed['depth'],
    "dependency": dependency['depth']
}

depth_distribution(depth, "imgs/depth.png", "Depth")

leaves = {
    "nested": nested['leaves'],
    "constituency": constituency['leaves'],
    "mixed": mixed['leaves'],
    "dependency": dependency['leaves']
}

leaves_distribution(leaves, "imgs/leaves.png", "Leaves")

symmetry = {
    "nested": nested['symmetry'],
    "constituency": constituency['symmetry'],
    "mixed": mixed['symmetry'],
    "dependency": dependency['symmetry']
}

symmetry_distribution(symmetry, "imgs/symmetry.png", "Symmetry")

branching = {
    "nested": nested['branching']['avg'],
    "constituency": constituency['branching']['avg'],
    "mixed": mixed['branching']['avg'],
    "dependency": dependency['branching']['avg']
}

branch_distribution(branching, "imgs/branching.png", "Branch")


