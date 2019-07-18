"""Package setup script."""
import sys

import setuptools

setuptools.setup(
    name="pilco",
    version="0.1",
    description="Implementation of PILCO",
    author="Eric Langlois",
    author_email="edl@cs.toronto.edu",
    license="MIT",
    packages=setuptools.find_packages(),
    install_requires=[
        "gpflow>=1.3.0",
        "gym>=0.7.4",
        "matplotlib>=3.0.1",
        "mujoco-py>=0.5.7",
        "numpy>=1.15.3",
        "progressbar2>=3.38.0",
        "pyyaml>=3.13",
        "scikit-learn>=0.20.0",
        "progressbar",
        "dataclasses",
    ],
    extras_require={
        "tf": ["tensorflow>=1.12.0,<2.0.0", "tensorflow-probability>=0.7.0"],
        "tf_gpu": ["tensorflow-gpu>=1.12.0,<2.0.0", "tensorflow-probability>=0.7.0"],
        "qt5": ["pyqt5>=5.12"],
    },
    setup_requires=["pytest-runner"],
    tests_require=["pytest>=3.4.1"],
    test_suite="pytest",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

target_version = (3, 7, 0)
if sys.version_info < target_version:
    print("WARNING:")
    print(
        "Package is designed for Python at least {}\n"
        "but you are using Python {}.".format(
            ".".join(str(x) for x in target_version),
            ".".join(str(x) for x in sys.version_info),
        )
    )
