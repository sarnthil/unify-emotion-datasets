# Requirements:

## System packages

- `Python 3.6+`
- `git`
## Installing Python dependencies

- `pip3 install requests sh`
- `git clone git@github.com:sarnthil/unify-emotion-datasets.git`


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

# Paper
[An Analysis of Annotated Corpora for Emotion Classification in Text](http://aclweb.org/anthology/C18-1179.pdf)


# Reference 
If you plan to use this corpus, please use this citation:

```
@inproceedings{Bostan2018,
  author = {Bostan, Laura Ana Maria and Klinger, Roman},
  title = {An Analysis of Annotated Corpora for Emotion Classification in Text},
  booktitle = {Proceedings of the 27th International Conference on Computational Linguistics},
  year = {2018},
  publisher = {Association for Computational Linguistics},
  pages = {2104--2119},
  location = {Santa Fe, New Mexico, USA},
  url = {http://aclweb.org/anthology/C18-1179},
  pdf = {http://aclweb.org/anthology/C18-1179.pdf}
}
```
