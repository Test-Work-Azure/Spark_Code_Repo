import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sparkpack", 
    version="0.0.1",
    author="Adarsh_Ayush",
    author_email="ayush.oturkar@fractal.ai",
    description="A small example package for ML end to end on spark using MLLib",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Test-Work-Azure/Spark_Code_Repo",
    packages=["sparkpack"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)


