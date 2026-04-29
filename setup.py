from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="bessopt",
    version="0.1.2",
    description="BESS optimisation: day-ahead and intraday battery dispatch using MILP",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="clarkmaio",
    python_requires=">=3.9",
    packages=["bessopt", "bessopt.webpage"],
    install_requires=[
        "loguru",
        "matplotlib",
        "polars",
        "cvxpy",
        "pimpmyplot",
    ],
    extras_require={
        "data": ["python-dotenv", "entsoe-py"],
        "dev": ["pytest"],
        "web": ["python-fasthtml", "fastapi", "uvicorn", "python-entsoe"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
    ],
)
