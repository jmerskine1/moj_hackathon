import json
import os
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import pandas as pd

import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend


def load_scores_from_json(filepath, section="intrasentence"):
    with open(filepath, "r") as f:
        data = json.load(f)
    
    if section not in data:
        raise ValueError(f"Section '{section}' not found in {filepath}")
    
    scores = [entry["score"] for entry in data[section]]
    return scores

def collect_data(json_files, section):
    all_data = []

    for path in json_files:
        model_name = os.path.basename(path).replace(".json", "")
        scores = load_scores_from_json(path, section=section)

        for score in scores:
            all_data.append({"Model": model_name, "Score": score})

    return pd.DataFrame(all_data)

def plot_violin(df, output_path=None):
    
    plt.figure(figsize=(10, 6))
    sns.violinplot(x="Model", y="Score", data=df, inner="box", density_norm="width", palette="muted")
    
    plt.title("Bias Scores Distribution by Model")
    plt.xticks(rotation=45)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"Saved plot to {output_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot StereoSet bias scores")
    parser.add_argument("json_files", nargs="+", help="Paths to one or more JSON result files")
    parser.add_argument("--section", default="intrasentence", help="Bias section: intrasentence or intersentence")
    parser.add_argument("--output", help="Path to save the plot (optional)")

    args = parser.parse_args()
    df = collect_data(args.json_files, args.section)
    plot_violin(df, args.output)

if __name__ == "__main__":
    main()
