from setuptools import setup, find_packages

setup(
    name="spike_tools",
    version="0.0.1",
    author="Monica Liu",
    author_email="mfliu@uw.edu",
    description="Access functions and tools for WCST spiking data",
    url="https://github.com/learning-2-learn/spike_tools",
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=["numpy>=1.8.1", "matplotlib", "pandas", "aiobotocore>=1.0.1", "h5py", "s3fs", "tqdm", "dask", "dask_gateway"]
)
