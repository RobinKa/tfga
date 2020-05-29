import subprocess
import os


def main():
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    def _call_pytest(name, *paths, env=None):
        subprocess.call(
            [
                "pytest", *paths,
                "--benchmark-warmup", "on",
                "--benchmark-json", os.path.join(results_dir, "%s.json" % name)
            ],
            env=env
        )

    _call_pytest("clifford", "test_clifford.py::test_clifford_add_mv_mv", "test_clifford.py::test_clifford_mul_mv_mv")
    _call_pytest("tfga-gpu", "test_tfga.py::test_tfga_add_mv_mv", "test_tfga.py::test_tfga_mul_mv_mv")

    cpu_env = os.environ.copy()
    cpu_env["CUDA_VISIBLE_DEVICES"] = "-1"
    _call_pytest("tfga-cpu", "test_tfga.py::test_tfga_add_mv_mv", "test_tfga.py::test_tfga_mul_mv_mv", env=cpu_env)


if __name__ == "__main__":
    main()
