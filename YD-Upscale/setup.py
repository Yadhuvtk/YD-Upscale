from setuptools import setup, find_packages

setup(
    name="yd_upscale",
    version="0.1.0",
    author="Yadhukrishna",
    description="Clean, modular, production-ready AI image upscaler project for anime, illustration, and text-preserving artwork.",
    packages=find_packages(exclude=["runtime", "runtime.*", "tests", "tests.*"]),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.21.0",
        "pyyaml>=6.0",
        "tqdm>=4.64.0",
        "opencv-python>=4.5.3",
        "pillow>=9.0.0",
        "tensorboard>=2.10.0",
        "einops>=0.6.0",
        "scikit-image>=0.19.0",
        "lpips>=0.1.4"
    ],
)
