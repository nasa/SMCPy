import re
from setuptools import setup


def get_property(prop, project):
    result = re.search(
        r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop),
        open(project + "/__init__.py").read(),
    )
    return result.group(1)


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="smcpy",
    version=get_property("__version__", "smcpy"),
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
    python_requires="~=3.4",
    classifiers=[
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
    ],
)
