from pathlib import Path

import pkg_resources
from setuptools import find_packages, setup

this_directory = Path(__file__).parent
VERSION = "1.0.0-dev9"
DESCRIPTION = (
    "A python package Library which implement IA algorithms to detect cracks and failures on roads. "
    "The package is wrapper around all the models and provides an interfaces to use them properly"
)

LONG_DESCRIPTION = (this_directory / "README.md").read_text()


with Path("requirements.txt").open() as requirements_txt:
    install_requires = [str(requirement) for requirement in pkg_resources.parse_requirements(requirements_txt)]

setup(
    # the name must match the folder name 'verysimplemodule'
    name="pavimentados",
    version=VERSION,
    author="Jose Maria Marquez Blanco",
    author_email="jose.marquez.blanco@gmail.com",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=install_requires,
    keywords=["Machine Learning", "crack detections", "computer vision", "safe road"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    include_package_data=True,
)
