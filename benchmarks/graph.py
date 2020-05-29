from glob import glob
import os
import json
from matplotlib import pyplot as plt
import numpy as np
import tfga
import clifford


def load_results(path):
    with open(path, "r", encoding="utf-8") as result_file:
        data = json.load(result_file)

    benchmarks = data["benchmarks"]

    mul_mv_mv_means = []
    mul_mv_mv_stds = []
    mul_mv_mv_num_elems = []

    add_mv_mv_means = []
    add_mv_mv_stds = []
    add_mv_mv_num_elems = []

    for benchmark in benchmarks:
        if "mul_mv_mv" in benchmark["name"]:
            mul_mv_mv_means.append(benchmark["stats"]["mean"])
            mul_mv_mv_stds.append(benchmark["stats"]["stddev"])
            mul_mv_mv_num_elems.append(
                benchmark["params"]["num_elements"])
        elif "add_mv_mv" in benchmark["name"]:
            add_mv_mv_means.append(benchmark["stats"]["mean"])
            add_mv_mv_stds.append(benchmark["stats"]["stddev"])
            add_mv_mv_num_elems.append(
                benchmark["params"]["num_elements"])

    return {
        "mul_mv_mv_means": np.array(mul_mv_mv_means, dtype=np.float64),
        "mul_mv_mv_stds": np.array(mul_mv_mv_stds, dtype=np.float64),
        "mul_mv_mv_num_elems": np.array(mul_mv_mv_num_elems, dtype=np.int32),
        "add_mv_mv_means": np.array(add_mv_mv_means, dtype=np.float64),
        "add_mv_mv_stds": np.array(add_mv_mv_stds, dtype=np.float64),
        "add_mv_mv_num_elems": np.array(add_mv_mv_num_elems, dtype=np.int32)
    }


def main():
    result_paths = glob(os.path.join("results", "*.json"))

    all_results = {
        os.path.splitext(os.path.basename(path))[0]: load_results(path) for path in result_paths
    }

    with plt.style.context("seaborn-darkgrid"):
        plt.figure(figsize=(6, 4))
        for name, results in all_results.items():
            plt.plot(results["mul_mv_mv_num_elems"], results["mul_mv_mv_means"], label=name)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Number of elements")
        plt.ylabel("Runtime [s]")
        plt.title("Elementwise geometric product of multivector batch")
        plt.legend()
        plt.savefig("results/mul_mv_mv.svg")

        plt.figure(figsize=(6, 4))
        for name, results in all_results.items():
            plt.plot(results["add_mv_mv_num_elems"], results["add_mv_mv_means"], label=name)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Number of elements")
        plt.ylabel("Runtime [s]")
        plt.title("Elementwise addition of multivector batch")
        plt.legend()
        plt.savefig("results/add_mv_mv.svg")


if __name__ == "__main__":
    main()
