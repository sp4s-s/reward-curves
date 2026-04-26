from setuptools import find_packages, setup

setup(
    name="curvature_dpo",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=[
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "hydra-core",
        "omegaconf",
        "einops",
    ],
    extras_require={
        "gpu": ["bitsandbytes", "flash-attn"],
        "dev": ["pytest", "ruff", "black"],
        "research": ["ray[train,tune]", "wandb", "modal", "triton"],
    },
)
