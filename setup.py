import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="textdsets",
    version="0.0.1",
    author="Valentin Liévin",
    author_email="valentin.lievin@gmail.com",
    description="Text Datasets for Pytorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vlievin/textdsets",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
