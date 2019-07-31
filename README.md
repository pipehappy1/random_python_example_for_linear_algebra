Download miniconda from the page:

    https://docs.conda.io/en/latest/miniconda.html

Open the shell and install the miniconda (change the script file name accordingly):

    bash Miniconda3-latest-MacOSX-x86_64.sh

Make sure you are in the project root directory before run the following command.
Create the conda environment:

    conda env create -f environment.yml

Activate the project environment:

    . activate python_example_linear_algegra_book

Install more libraries:

    conda install -y --file requirements.txt

Go to each chapX directory to run those scripts.
