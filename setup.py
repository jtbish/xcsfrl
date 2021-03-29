import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="xcsfrl",
    version="0.0.1",
    author="Jordan T. Bishop",
    author_email="jordanbishop26@gmail.com",
    description="XCSF for Reinforcement Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jtbish/xcsfrl",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy",
        "gym"
    ],
    python_requires='>=3.6',
)
