import setuptools

with open("README.md", "r") as fh:
    description = fh.read()

setuptools.setup(
    name="ihrismart-healthit",
    version="0.0.1",
    author="Duncan",
    author_email="all@healthit.uonbi.ac.ke",
    description="Creates a datamart for hr data",
    long_description=description,
    long_description_content_type="text/markdown",
    url="https://github.com/ndiithi/ihrismart",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU GENERAL PUBLIC LICENSE",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)