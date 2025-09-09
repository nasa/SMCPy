import os
from setuptools import setup

setup(
    name=os.environ.get("PACKAGE_NAME", "smcpy"),
)
