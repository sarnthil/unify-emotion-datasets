# Requirements:

## System packages

- `Python 3.6`
- `git`
## Installing Python dependencies

- `pip3 install requests sh`
- `git clone git@bitbucket.org:laurabostan/unify-emotion-datasets.git`


This will create a new folder called `unify-emotion-datasets`.

# Running the two scripts

First run the script that downloads all obtainable datasets:

- `cd unify-emotion-datasets  # go inside the repository`
- `python3 download_datasets.py`


Please read carefully the instructions, you will be asked to read and confirm having read the licenses and terms of use of each dataset. 
In case the dataset is not obtainable directly you will be given instructions on how to obtain the dataset.

Then run the script that unifies the downloaded datasets, which will be located in `unify-emotion-datasets/datasets/`:


`python3 create_unified_datasets.py`


This will create a new file called `unified-dataset.jsonl` in the same folder.

Also, we advise you to cite the papers corresponding to the datasets you use.
The corresponding `bibtex` citations you find in the file `datasets/README.md` or while
running `download_datasets.py`. 

# Reference 
If you plan to use this corpus, please cite it as follows: 

Laura Ana Maria Bostan and Roman Klinger. A survey on annotated data sets for emotion classification in text. In Proceedings of COLING 2018, the 27th International Conference on Computational Linguistics, Santa Fe, USA, August 2018.
