from setuptools import setup, find_packages

setup(
    name="robolab",
    packages=find_packages(),
    version="1.0.0",
    install_requires=[
        "joblib>=1.2.0",
        "prettytable",
        "pyyaml",
    ],
)
