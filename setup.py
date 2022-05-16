from setuptools import setup, find_packages

setup(
    name="TransforKmers",
    version="1.0",
    description=(
        "A task-agnostic facility to pretrain/finetune/use a"
        "Transformer based model to classify your biosequences."
    ),
    author="Matthias Lorthiois",
    author_email="matthias.lorthiois@cnrs.fr",
    url="https://github.com/mlorthiois/transforkmers",
    packages=find_packages(),
    entry_points={
        "console_scripts": ["transforkmers=transforkmers.commands:main"],
    },
    install_requires=[
        "transformers>=4.18",
    ],
    python_requires=">=3.8",
)
