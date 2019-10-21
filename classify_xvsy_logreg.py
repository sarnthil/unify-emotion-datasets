#!/usr/bin/env python3
"""Classify using MaxEnt algorithm

Usage:
    classify_xvsy_logreg.py [options] <first> <second>
    classify_xvsy_logreg.py [options] --all-vs <second>

Options:
    -j --json=<JSONFILE>  Filename of the json file [default: unified-dataset.jsonl]
    -a --all-vs<=dataset> Dataset name of the testing data
    -d --debug            Use a small word list and a fast classifier
    -o --output=<OUTPUT>  Output folder [default: .]
    -m --force-multi      Force using multi-label classification
    -k --keep-last        Quit immediately if results file found

"""
import re
import sys
import os
import json
import random
import math
import operator as op

from collections import Counter, defaultdict, namedtuple

import docopt
import numpy as np

from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from sklearn.metrics import classification_report
from sklearn.externals import joblib

np.random.seed(0)
random.seed(0)


Report = namedtuple(
    "Report", ["precision", "recall", "accuracy", "f1", "tp", "tn", "fp", "fn"]
)

PATTERN_TOKENS = re.compile(r"[a-z]+")


def cheatydiv(x, y):
    return math.nan if y == 0 else x / y


def get_labels(train, test, operation=op.and_, mode="multi"):
    """Return a list of the emotional intersection of two sources."""
    emotions = set()
    if mode == "single":
        emotions.add("noemo")
    train_emotions = set(
        emotion
        for data in train
        for emotion in data["emotions"]
        if data["emotions"][emotion] is not None
    )
    # print(train_emotions)
    test_emotions = set(
        emotion
        for emotion in test[0]["emotions"]
        if test[0]["emotions"][emotion] is not None
    )
    # print(test_emotions)
    return list(emotions | operation(train_emotions, test_emotions))


def get_emotion(emovals, labels, emotions, mode="multi"):
    if mode == "single":
        truthy = len(list(filter(bool, emovals.values())))
        if truthy == 1:
            emotion = [v for v in emovals if emovals[v]][0]
        elif truthy == 0:
            emotion = "noemo"
        else:
            # emotion = sorted(
            #     ((k, v) for k, v in emovals.items() if v),
            #     key=lambda x: x[1],
            #     reverse=True,
            # )[0][0]
            raise ValueError("Dataset marked as 'single' contains multiple emotions")
        return emotions.get(emotion, emotions.get("noemo"))
    else:
        el = [int((emovals[label] or 0) > 0.1) for label in labels]
        return np.array(el)


def get_vector(text, wordlist):
    tokens = set(tokenize(text))
    return [1 if word in tokens else 0 for word in wordlist]


def make_arrays(train, test, words, labels, mode="multi", all_vs=False):
    emotions = {label: x for x, label in enumerate(labels)}
    train_x, train_y, test_x, test_y = [], [], [], []

    # debug_train_emos = Counter()
    for data in train:
        # debug_train_emos[get_emotion(data["emotions"], emotions)] += 1
        # Discard examples where we don't have all selected emotions
        if (
            mode == "single"
            or all_vs
            or all(data["emotions"][emo] is not None for emo in labels)
        ):
            train_y.append(get_emotion(data["emotions"], labels, emotions, mode))
            train_x.append(get_vector(data["text"], words))

    # debug_test_emos = Counter()
    for data in test:
        # debug_test_emos[get_emotion(data["emotions"], emotions)] += 1
        test_y.append(get_emotion(data["emotions"], labels, emotions, mode))
        test_x.append(get_vector(data["text"], words))

    # print("DEBUG counts")
    # print(debug_train_emos)
    # print(debug_test_emos)
    # print("----")

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    return train_x, train_y, test_x, test_y


def filtered_texts(filename, source):
    with open(filename) as f:
        for line in f:
            data = json.loads(line)
            if data["source"] == source:
                yield data["text"]

def classification_report_own_single(test_y, predict_y, labels):
    reports = {}
    num2emo = {i: label for i, label in enumerate(labels)}
    decisions = defaultdict(Counter)
    for t, p in zip(test_y, predict_y):
        decisions[t][p] += 1
    for label in decisions:
        tp = decisions[label][label]
        fp = sum(decisions[x][label] for x in decisions if x != label)
        tn = sum(
            decisions[x][y]
            for x in decisions
            for y in decisions[x]
            if x != label and y != label
        )
        fn = sum(decisions[label][y] for y in decisions[label] if y != label)
        precision = tp / (tp + fp) if tp + fp else math.nan
        recall = tp / (tp + fn) if tp + fn else math.nan
        f1 = 2 * cheatydiv((precision * recall), (precision + recall))
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        reports[num2emo[label]] = Report(
            precision, recall, accuracy, f1, tp, tn, fp, fn
        )
    return reports


def classification_report_own_multi(test_y, predict_y, labels):
    reports = {}
    num2emo = {i: label for i, label in enumerate(labels)}
    emo2num = {label: i for i, label in enumerate(labels)}
    decisions = defaultdict(Counter)
    for label in labels:
        tp = fp = tn = fn = 0
        for t, p in zip(test_y, predict_y):
            # decisions[t][p] += 1
            tp += bool(t[emo2num[label]] and p[emo2num[label]])
            fp += bool(p[emo2num[label]] and not t[emo2num[label]])
            fn += bool(t[emo2num[label]] and not p[emo2num[label]])
            tn += bool(not t[emo2num[label]] and not p[emo2num[label]])
        precision = tp / (tp + fp) if tp + fp else math.nan
        recall = tp / (tp + fn) if tp + fn else math.nan
        f1 = 2 * cheatydiv((precision * recall), (precision + recall))
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        reports[label] = Report(precision, recall, accuracy, f1, tp, tn, fp, fn)
    return reports


def analyse_results(test_y, predict_y, labels, test, first, second, output, mode):
    prefix = f"{first}_vs_{second}_{mode}"
    fprefix = output + "/" + prefix
    with open(fprefix + ".txt", "w") as f, open(fprefix + ".json", "w") as g:
        # print(confusion_matrix(test_y, predict_y), file=f)
        prec, reca, f1, supp = precision_recall_fscore_support(
            test_y, predict_y, pos_label=None, average="micro"
        )
        accuracy = accuracy_score(test_y, predict_y)
        for score, name in [
            (prec, "Precision"),
            (reca, "Recall"),
            (f1, "F1-score"),
            (accuracy, "Accuracy"),
        ]:
            print(name, score, sep="\t", file=f)

        # print("real:", Counter(test_y), file=f)
        # print("predicted:", Counter(predict_y), file=f)

        print(test_y[:10], predict_y[:10], file=f)
        emotions = {i: label for i, label in enumerate(labels)}
        for text, real, predicted, _ in zip(test, test_y, predict_y, range(20)):
            if mode == "multi" and np.array_equal(real, predicted):
                continue
            elif mode == "single" and real == predicted:
                continue
            print(text, "=> predicted:", predicted, ", truth:", real, file=f)
        if mode == "multi":
            results = classification_report_own_multi(test_y, predict_y, labels)
        elif mode == "single":
            results = classification_report_own_single(test_y, predict_y, labels)
        json.dump(
            {
                "precision": prec,
                "recall": reca,
                "f1": f1,
                "accuracy": accuracy,
                "name": prefix,
                **{
                    (emotion + "_" + metric): getattr(results[emotion], metric)
                    for emotion in results
                    for metric in Report._fields
                },
            },
            g,
        )
        g.write("\n")
        # for result in results:
        #     print("Results for", result, file=f)
        #     print(
        #         "\n".join(
        #             key + " => " + str(
        #                 getattr(results[result], key)
        #             ) for key in Report._fields
        #         ),
        #         file=f,
        #     )
        # print(classification_report(test_y, predict_y, labels), file=f)


def tokenize(text):
    return re.findall(r"\p{L}+", text.lower())


# this is bad. memory error for all_vs (too many words...)
def get_wordlist(dataset):
    """Get a bag of words from a dataset."""
    bag = set()
    for data in dataset:
        bag.update({token for token in tokenize(data["text"])})
    return list(bag)


# ask roman what would be a good vocab here?
def get_wordlist_debug(dataset):
    """Get a bag of words from a dataset."""
    bag = Counter()
    for data in dataset:
        bag.update({token for token in tokenize(data["text"])})
    return list(map(op.itemgetter(0), bag.most_common(5000)))


def hacky_train_test_split(training, train_size=0.8, first=None, second=None):
    tra, tes = [], []
    for example in training:
        if example.get("split") == "train" or example["source"] != second:
            tra.append(example)
        elif example.get("split") == "test":
            tes.append(example)
        else:
            # don't try this at home
            [tes, tra][random.random()<train_size].append(example)
    return tra, tes


def get_train_test(jsonfile, train, test):
    same = test in train.split(",")
    training, testing = [], []
    with open(jsonfile) as f:
        for line in f:
            data = json.loads(line)
            if data["source"] in train.split(",") or (train is None and data["source"] != test):
                training.append(data)
            elif data["source"] == test:
                testing.append(data)
    if same:
        training, testing = hacky_train_test_split(training, train_size=0.8, first=train, second=test)
    return training, testing


def get_clf_mode(train, test):
    """ Detect whether we are in single-label to single-label mode or not. """
    first = "single"
    for example in train:
        if example.get("labeled", "multi") == "multi":
            first = "multi"
    for example in test:
        if example.get("labeled", "multi") == "multi":
            return first, "multi"
    return first, "single"


if __name__ == "__main__":
    args = docopt.docopt(__doc__, version="0.0.1")
    print(args)
    print("Getting data")
    training_data, testing_data = get_train_test(
        args["--json"],
        args["<first>"] if not args["--all-vs"] else None,
        args["<second>"],
    )
    first, second = (
        ["multi", "multi"] if args["--force-multi"] else get_clf_mode(training_data, testing_data)
    )
    mode = "multi" if "multi" in [first, second] else "single"
    if (
        os.path.exists(
            "{}/{}_vs_{}_{}.json".format(
                args["--output"], args["<first>"], args["<second>"], mode
            )
        )
        and args["--keep-last"]
    ):
        print("We already have results for this; quitting")
        sys.exit(0)

    print("Detected mode: {}...".format(mode))
    print(len(training_data), len(testing_data))
    print("Getting wordlist...")
    if args["--debug"]:
        wordlist = get_wordlist_debug(training_data)
    else:
        wordlist = get_wordlist_debug(training_data)
        # wordlist = get_wordlist(training_data)
    print("Getting emotions")
    labels = get_labels(training_data, testing_data, mode=mode)
    print(labels)
    print("Making arrays")
    train_x, train_y, test_x, test_y = make_arrays(
        training_data, testing_data, wordlist, labels, mode, args["--all-vs"]
    )
    if any(not part.size for part in [train_x, train_y, test_x, test_y]):
        print("Train or test empty. Did you misspell the dataset name?")
        sys.exit(1)
    print("Initializing classifier")
    if args["--debug"]:
        classifier = RandomForestClassifier()
    elif mode == "single":
        classifier = LogisticRegressionCV(
            cv=10,
            penalty="l2",
            fit_intercept=True,
            solver="sag",
            scoring="f1",
            refit=True,
            # n_jobs=-1,
            class_weight="balanced",
        )
    else:
        classifier = OneVsRestClassifier(
            LogisticRegressionCV(
                cv=10,
                penalty="l2",
                fit_intercept=True,
                solver="sag",
                scoring="f1",
                refit=True,
                class_weight="balanced",
                tol = 0.1,
            ),
            n_jobs=-1,
        )

    print("Training...")
    classifier.fit(train_x, train_y)
    print("Predicting...")
    if first == "multi" and second == "single":
        predict_y = classifier.predict_proba(test_x)
        helper = np.zeros_like(predict_y)
        helper[range(len(predict_y)), predict_y.argmax(1)] = 1
        predict_y = helper
    else:
        predict_y = classifier.predict(test_x)

    print("Analysing...")
    if args["--all-vs"]:
        args["<first>"] = "all"
    analyse_results(
        test_y,
        predict_y,
        labels,
        testing_data,
        args["<first>"],
        args["<second>"],
        args["--output"],
        mode,  # TODO
    )
    joblib.dump(classifier, "classifier.pkl")
