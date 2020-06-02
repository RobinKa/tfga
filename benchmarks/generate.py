import subprocess
import os


def main():
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    def _call_pytest(name, *paths, env=None):
        subprocess.call(
            [
                "pytest", *paths,
                "--benchmark-json", os.path.join(results_dir, "%s.json" % name)
            ],
            env=env
        )

    # Test tfga (default on gpu if available)
    for i in [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000]:
        _call_pytest("tfga-gpu_mul-mv-mv_%d" % i, "test_tfga.py::test_tfga_mul_mv_mv[%d]" % i)

    # Test tfga cpu (hide cuda gpus)
    cpu_env = os.environ.copy()
    cpu_env["CUDA_VISIBLE_DEVICES"] = "-1"
    for i in [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000]:
        _call_pytest("tfga_mul-mv-mv_%d" % i, "test_tfga.py::test_tfga_mul_mv_mv[%d]" % i, env=cpu_env)

    # Test clifford
    for i in [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000]:
        _call_pytest("clifford_mul-mv-mv_%d" % i, "test_tfga.py::test_clifford_mul_mv_mv[%d]" % i)

    # Test clifford raw
    for i in [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000]:
        _call_pytest("clifford-raw_mul-mv-mv_%d" % i, "test_tfga.py::test_clifford_raw_mul_mv_mv[%d]" % i)

if __name__ == "__main__":
    main()
