from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="poscidyn",
    version="0.1.0",
    description="Identification and simulation of nonlinear oscillators dynamics.",
    author="Raymond Knetemann",
    author_email="rknetemann@student.tudelft.nl",
    url="https://github.com/rknetemann/poscidyn",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)