from setuptools import setup


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="smcpy",
    version='0.1.6',
    author="Patrick Leser",
    author_email="patrick.e.leser@nasa.gov",
    description="A package for performing uncertainty quantification using a parallelized sequential Monte Carlo sampler.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nasa/SMCPy",
    packages=["smcpy", "smcpy.smc", "smcpy.mcmc", "smcpy.utils"],
    install_requires=[
        "numpy",
        "h5py",
        "matplotlib",
        "tqdm",
        "scipy",
        "pandas",
        "seaborn",
        "pytest",
        "pytest-mock",
    ],
    python_requires=">=3.6",
    license="NOSA v1.3",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
    ],
)
