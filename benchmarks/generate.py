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

    def _run_parameterized(lib_name, fn_name, fn_path, num_elements, env=None):
        for i in num_elements:
            _call_pytest("%s_%s_%d" % (lib_name, fn_name, i), fn_path % i, env=env)

    num_elements = [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000]

    cpu_env = os.environ.copy()
    cpu_env["CUDA_VISIBLE_DEVICES"] = "-1"

    # Multiply multivector batches
    _run_parameterized("tfga-gpu", "mul-mv-mv", "test_tfga.py::test_tfga_mul_mv_mv[%d]", num_elements)
    _run_parameterized("tfga", "mul-mv-mv", "test_tfga.py::test_tfga_mul_mv_mv[%d]", num_elements, env=cpu_env)
    _run_parameterized("clifford", "mul-mv-mv", "test_clifford.py::test_clifford_mul_mv_mv[%d]", num_elements)
    _run_parameterized("clifford-raw", "mul-mv-mv", "test_clifford.py::test_clifford_raw_mul_mv_mv[%d]", num_elements)

    # Add multivector batches
    _run_parameterized("tfga-gpu", "add-mv-mv", "test_tfga.py::test_tfga_add_mv_mv[%d]", num_elements)
    _run_parameterized("tfga", "add-mv-mv", "test_tfga.py::test_tfga_add_mv_mv[%d]", num_elements, env=cpu_env)
    _run_parameterized("clifford", "add-mv-mv", "test_clifford.py::test_clifford_add_mv_mv[%d]", num_elements)
    _run_parameterized("clifford-raw", "add-mv-mv", "test_clifford.py::test_clifford_raw_add_mv_mv[%d]", num_elements)

if __name__ == "__main__":
    main()
