from setuptools import setup, find_packages

setup(
    name="radial_schrodinger",
    version="1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=["numpy", "scipy", "matplotlib", "seaborn"],
)
