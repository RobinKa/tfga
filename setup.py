from setuptools import setup

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
    version="0.1",
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
