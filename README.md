# Requirements:

## System packages

- `Python 3.6+`
- `git`

## Installing Python dependencies

- `pip3 install requests sh click` 
- `pip3 install regex docopt numpy sklearn scipy`, if you want to use `classify_xvsy_logreg.py`
- `git clone git@github.com:sarnthil/unify-emotion-datasets.git`


This will create a new folder called `unify-emotion-datasets`.

# Running the two scripts

First run the script that downloads all obtainable datasets:

- `cd unify-emotion-datasets  # go inside the repository`
- `python3 download_datasets.py`


Please read carefully the instructions, you will be asked to read and confirm having read the licenses and terms of use of each dataset. 
In case the dataset is not obtainable directly you will be given instructions on how to obtain the dataset.

Then run the script that unifies the downloaded datasets, which will be located in `unify-emotion-datasets/datasets/`:

`python3 create_unified_dataset.py`


This will create a new file called `unified-dataset.jsonl` in the same folder.

Also, we advise you to cite the papers corresponding to the datasets you use.
The corresponding `bibtex` citations you find in the file `datasets/README.md` or while
running `download_datasets.py`. 

# Paper
[An Analysis of Annotated Corpora for Emotion Classification in Text](http://aclweb.org/anthology/C18-1179.pdf)
If you want to reuse the code for the emotion classification task, see the script `classify_xvsy_logreg.py`:

 `python3 classify_xvsy_logreg.py --help` will show you the following: 

``` 
Classify using MaxEnt algorithm

Usage:
    classify_xvsy_logreg.py [options] <first> <second>
    classify_xvsy_logreg.py [options] --all-vs <second>

Options:
    -j --json=<JSONFILE>  Filename of the json file [default: ../unified.jsonl]
    -a --all-vs<=dataset> Dataset name of the testing data
    -d --debug            Use a small word list and a fast classifier
    -o --output=<OUTPUT>  Output folder [default: .]
    -m --force-multi      Force using multi-label classification
    -k --keep-last        Quit immediately if results file found
```
For example if you want to train on TEC and test on SSEC do the following:

    python3 classify_xvsy_logreg.py -d tec emoint 

The names of the dataset are the ones used in the file `unified-dataset.jsonl` in the field `source`.

# Tip
Use [`jq`](https://stedolan.github.io/jq/manual/) for an easy interaction with the `unified-dataset.jsonl`

Examples of how to use it for various tasks:
- selecting the instances of that have as a source `crowdflower` or `tec`
 `jq 'select(.source=="crowdflower" or .source =="tec")' <unified-dataset.jsonl | less `
- count how often instances are annotated with high surprise per dataset
`jq 'select(.emotions.surprise >0.5) | .source' <unified-dataset.jsonl | sort | uniq -c`   

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

