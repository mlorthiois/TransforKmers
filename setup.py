from setuptools import setup

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
    packages=["transforkmers"],
    entry_points={
        "console_scripts": ["transforkmers=commands:main"],
    },
)
