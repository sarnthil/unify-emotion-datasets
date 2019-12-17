import json
import os
from glob import glob
from sklearn.model_selection import train_test_split
import regex
import csv
from collections import defaultdict

TOKENS = regex.compile(r"\p{L}+|[.?!]")

path = "/Users/sarnthil/seat/fastai/datasets/"


def tokenize(text):
    return " ".join(TOKENS.findall(text))


# def create_dataset(dataset_source):
def create_dataset():
    examples = []
    examples_by_source = defaultdict(list)
    with open("unified-dataset.jsonl") as f:
        for line in f:
            datum = json.loads(line)
            source = datum["source"]
            # if datum["source"] != dataset_source:
            # continue
            # examples.append((map_emotion(datum), tokenize(datum["text"]), source))
            emotion = get_emotion(datum)
            text = tokenize(datum["text"])
            examples.append((emotion, text))
            examples_by_source[source].append((emotion, text))
        # train, test = train_test_split(examples)
    # with open(f"{dataset_source}.csv", "w") as f:
    with open("unified_without_source.csv", "w") as csvfile:
        fieldnames = ["label", "text"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for number, text in examples:
            if number is None or not text:
                continue
            writer.writerow({"label": number, "text": text})
        # for number, text in test:
        #     if number is None or not text:
        #         continue
        #     writer.writerow({"label": number, "text": text})
        # for number, text in train:
        #     if number is None or not text:
        #         continue
        #     writer.writerow({"label": number, "text": text})

    for source, examples in examples_by_source.items():
        with open(f"{source}.csv", "w") as csvfile:
            fieldnames = ["label", "text"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for number, text in examples:
                if number is None or not text:
                    continue
                writer.writerow({"label": number, "text": text})
            # for number, text in test:
            #     if number is None or not text:
            #         continue
            #     writer.writerow({"label": number, "text": text})
            # for number, text in train:
            #     if number is None or not text:
            #         continue
            #     writer.writerow({"label": number, "text": text})

# os.makedirs()

# with open("emo2id.json") as f:
#     emo2id = json.load(f)

emo2id = {
    "noemo": 0,
    "joy": 1,
    "anger": 2,
    "sadness": 3,
    "disgust": 4,
    "fear": 5,
    "trust": None,
    "surprise": 6,
    "love": None,
    "confusion": None,
    "anticipation": None,
    "shame": None,
    "guilt": None,
}

def emotion_val(datum):
    return [
        (
            (
                datum["emotions"][emo]
                if datum["emotions"][emo] is not None
                else 0
            ),
            emo,
        )
        for emo in datum["emotions"]
    ]

def map_emotion(datum):
    emo_val = emotion_val(datum)
    if sum(x[0] for x in emo_val) > 0:
        return emo2id[max(emo_val)[1]]
    else:
        return emo2id["noemo"]

def get_emotion(datum):
    emo_val = emotion_val(datum)
    if sum(x[0] for x in emo_val) > 0:
        return max(emo_val)[1]
    else:
        return "noemo"


if __name__ == "__main__":
    create_dataset()
