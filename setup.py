"""Installation script for the 'unitree_rl_mjlab' python package."""

from setuptools import setup, find_packages

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "mjlab==1.2.0",
    # "joblib",
    # "opencv-python",
    # "scikit-learn",
    # "coacd",
    # "torch",  # 由用户按 CUDA 版本自行安装（参考 README）
]

# Installation operation
setup(
    name="unitree_rl_mjlab",
    packages=find_packages(include=["src", "src.*"]),
    version="0.0.1",
    install_requires=INSTALL_REQUIRES,
)
