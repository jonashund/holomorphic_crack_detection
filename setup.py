from setuptools import setup, find_packages

setup(
    name="holomorphic crack detection",
    version="1.0",
    author="Nicolas Cuenca, Jonas Hund, Tito Andriollo",
    author_email="nicolas.cuenca@sigma-clermont.fr, jonas.hund@gmx.de, titoan@mpe.au.dk",
    description="Holomorphic neural network combined with a genetic algorithm approach to solve 2D crack detection problems. Based on the PIHNN framework by Matteo Calafà and further developed by Nicolas Cuenca, Jonas Hund, and Tito Andriollo.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/jonashund/holomorphic_crack_detection",
    packages=find_packages(),
    install_requires=[
        "matplotlib>=3.5.0",
        "numpy>=1.22.0",
        "scipy>=1.9.0",
        "torch",
        "tqdm",
        "pihnn @ git+https://github.com/teocala/pihnn.git",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
