from setuptools import setup
import os

# Read README.md
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as readme_file:
    long_description = readme_file.read()

setup(
    name="tfga",
    packages=["tfga"],
    extras_require={
        "tf": ["tensorflow>=2.0.0"],
        "tf_gpu": ["tensorflow-gpu>=2.0.0"],
        "tf_nightly": ["tf-nightly>=2.0.0"],
        "tf_nightly_gpu": ["tf-nightly-gpu>=2.0.0"],
    },
    description="Clifford and Geometric Algebra with TensorFlow",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version="0.1.11",
    url="https://github.com/RobinKa/tfga",
    author="Robin 'Tora' Kahlow",
    author_email="tora@warlock.ai",
    license="MIT",
    keywords="geometric-algebra clifford-algebra tensorflow multi-vector para-vector mathematics machine-learning",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Development Status :: 3 - Alpha",

        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3 :: Only",

        "Intended Audience :: Education",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",

        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",

        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ]
)
