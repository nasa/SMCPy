import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="SMCPy",
    version="0.1.0",
    author="Patrick Leser",
    author_email="patrick.e.leser@nasa.gov",
    description="Sequential Monte Carlo with Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nasa/SMCPy",
    packages=["smcpy",
              "smcpy.smc",
              "smcpy.particles",
              "smcpy.mcmc",
              "smcpy.model",
              "smcpy.utils",
              "smcpy.hdf5"],
    install_requires=['numpy', 'scipy', 'matplotlib', 'h5py', 'pymc', 'statsmodels', 'tqdm'],
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "Operating System :: OS Independent",
    ],
)
