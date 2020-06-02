from glob import glob
import os
import pandas as pd
import json
from matplotlib import pyplot as plt
import numpy as np
import clifford


def load_results(path):
    with open(path, "r", encoding="utf-8") as result_file:
        data = json.load(result_file)

    file_name = os.path.splitext(os.path.basename(path))[0]

    # file name: <lib>_<function>_<elements>
    lib_name, fn_name, num_elements = file_name.split("_")
    num_elements = int(num_elements)

    benchmarks = data["benchmarks"]

    assert len(benchmarks) == 1
    benchmark = benchmarks[0]

    mean, stddev = benchmark["stats"]["mean"], benchmark["stats"]["stddev"]

    return {
        "lib_name": lib_name,
        "fn_name": fn_name,
        "num_elements": num_elements,
        "mean": mean,
        "stddev": stddev
    }


def main():
    result_paths = sorted(glob(os.path.join("results", "*.json")))
    out_path = "output"

    os.makedirs(out_path, exist_ok=True)

    all_results = list(map(load_results, result_paths))

    df = pd.DataFrame(all_results)
    print(df)

    with plt.style.context("seaborn-darkgrid"):
        for fn_name, fn_df in df.groupby(by="fn_name"):
            plt.figure(figsize=(6, 4))
            for lib_name, lib_df in fn_df.groupby(by="lib_name"):
                plt.errorbar(lib_df["num_elements"], lib_df["mean"], lib_df["stddev"], label=lib_name)
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel("Number of elements")
            plt.ylabel("Runtime [s]")
            plt.title(fn_name)
            plt.legend()
            plt.savefig(os.path.join(out_path, "%s.svg" % fn_name))


if __name__ == "__main__":
    main()
