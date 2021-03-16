import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="SMCPy",
    version="1.1",
    author="Patrick Leser",
    author_email="patrick.e.leser@nasa.gov",
    description="Sequential Monte Carlo with Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nasa/SMCPy",
    packages=["smcpy",
              "smcpy.smc",
              "smcpy.mcmc",
              "smcpy.utils"],
    install_requires=['numpy', 'matplotlib', 'tqdm'],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Operating System :: OS Independent",
    ],
)
