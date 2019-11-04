import csv
import itertools
import json
import random
import re
import os.path as path
from collections import Counter
from glob import iglob


class WeirdlyEncodedFile:
    def __init__(self, fh):
        self.fh = fh

    @staticmethod
    def decode(something):
        try:
            return (
                something.decode("utf-8")
                .encode("latin-1")
                .decode("windows-1252")
            )
        except UnicodeEncodeError:
            return something.decode("utf-8")

    def readline(self):
        return self.decode(self.fh.readline())

    def read(self):
        return self.decode(self.fh.read())

    def __next__(self):
        return self.decode(next(self.fh))

    def __iter__(self):
        return self


def emotion_mapping(emotions, dataset_emotions):
    """
    Arguments: {"joy": 1, "sadness": 0.8}, ["joy", "sadness", "trust"]
    Returns: {"joy": 1, "sadness": 0.8, "trust": 0, "disgust": None, ...}
    """
    all_emotions = [
        "joy",
        "anger",
        "sadness",
        "disgust",
        "fear",
        "trust",
        "surprise",
        "love",
        "noemo",
        "confusion",
        "anticipation",
        "shame",
        "guilt",
    ]  # ALL of them; 11ish

    d = {emotion: None for emotion in all_emotions}
    for emotion in all_emotions:
        if emotion in dataset_emotions:
            d[emotion] = emotions.get(emotion, 0)
    return d


def extract_tec(folder):
    mapping = {
        "joy": "joy",
        "sadness": "sadness",
        "anger": "anger",
        "surprise": "surprise",
        "fear": "fear",
        "disgust": "disgust",
    }
    # only ekman basic 6

    emofile = path.join(folder, "Jan9-2012-tweets-clean.txt")
    with open(emofile) as e:
        for eline in e:
            emotion = eline.split("::")[1].strip()
            tweet = eline.split(":")[1].strip()
            emoname = mapping.get(emotion)
            d = emotion_mapping({emoname: 1}, mapping.values())
            yield {"source": "tec", "text": tweet, "emotions": d, "split": None}


def extract_jointMultitaskEmo(folder):
    mapping = {
        "joy": "joy",
        "sadness": "sadness",
        "anger": "anger",
        "surprise": "surprise",
        "fear": "fear",
        "disgust": "disgust",
        "anticipation": "anticipation",
        "noemotion": "noemo",
        "other": "noemo",
        "trust": "trust",
    }
    # Plutchik

    emofile = path.join(folder, "emotion_multigenre_corpus_setences.txt")
    with open(emofile) as e:
        for eline in e:
            emotion = eline.split("\t")[2].strip()
            text = eline.split("\t")[1].strip()
            emoname = mapping.get(emotion)
            d = emotion_mapping({emoname: 1}, mapping.values())
            yield {
                "source": "jointMultitaskEmo",
                "text": text,
                "emotions": d,
                "split": None,
            }


def extract_emoint(folder):
    mapping = {
        "joy": "joy",
        "sadness": "sadness",
        "anger": "anger",
        "fear": "fear",
    }

    emofile = path.join(folder, "emoint_all")
    with open(emofile) as e:
        for eline in e:
            emotion = eline.split("\t")[2].strip()
            tweet = eline.split("\t")[1].strip()
            emoname = mapping.get(emotion)
            d = emotion_mapping({emoname: 1}, mapping.values())
            yield {
                "source": "emoint",
                "text": tweet,
                "emotions": d,
                "split": None,
            }


def extract_electoraltweets(folder):
    subfolder1 = "Annotated-US2012-Election-Tweets/Questionnaire2/Batch1"
    subfolder2 = "Annotated-US2012-Election-Tweets/Questionnaire2/Batch2"

    mapping = {
        "amazement": "surprise",
        "anticipation": "anticipation",
        "expectancy": "anticipation",
        "interest": "anticipation",
        "anger": "anger",
        "annoyance": "anger",
        "apprehension": "fear",
        "calmness": "joy",
        "disappointment": "disgust",
        "disgust": "disgust",
        "dislike": "disgust",
        "elation": "joy",
        "fear": "fear",
        "fury": "anger",
        "gloominess": "sadness",
        "grief": "sadness",
        "happiness": "joy",
        "hate": "disgust",
        "hostility": "anger",
        "indifference": "disgust",
        "BLANK": "noemo",
        "acceptance": "trust",
        "joy": "joy",
        "like": "trust",
        "admiration": "trust",
        "vigilance": "anticipation",
        "panic": "fear",
        "sadness": "sadness",
        "surprise": "surprise",
        "serenity": "joy",
        "sorrow": "sadness",
        "terror": "fear",
        "trust": "trust",
        "confusion": "confusion",
        "uncertainty": "surprise",
        "indecision": "confusion",
    }

    emofile1 = path.join(folder, subfolder1, "AnnotatedTweets.txt")
    emofile2 = path.join(folder, subfolder2, "AnnotatedTweets.txt")

    for emofile in [emofile1, emofile2]:
        with open(emofile, newline="\n") as e:
            next(e)
            for eline in e:
                emotions = [
                    emo.strip()
                    for emo in eline.split("\t")[15].strip().split(" or ")
                ]
                tweet = eline.split("\t")[13].strip()
                emonames = [mapping.get(emotion) for emotion in emotions]
                if None in emonames:
                    print(emonames, emotions)
                d = emotion_mapping(
                    {emoname: 1 for emoname in emonames}, mapping.values()
                )
                yield {
                    "source": "electoraltweets",
                    "text": tweet,
                    "emotions": d,
                    "split": None,
                }


def extract_grounded_emotions(folder):
    subfolder = "GroundedEmotions/collected_data"
    mapping = {"happy": "joy", "sad": "sadness"}

    emofile = path.join(folder, subfolder, "collected_tweets.txt")
    text = path.join(folder, subfolder, "collected_user_history_data.txt")
    with open(emofile) as e, open(text) as t:
        for eline, tline in zip(e, t):
            emotion = eline.split("|")[2].strip()
            tweet = tline.split("|")[2].strip()
            emoname = mapping.get(emotion)
            d = emotion_mapping({emoname: 1}, mapping.values())
            yield {
                "source": "grounded_emotions",
                "text": tweet,
                "emotions": d,
                "split": None,
            }


def extract_isear(folder):
    # /* subfolder = "py_isear_dataset" */
    mapping = {
        "1": "joy",
        "2": "fear",
        "3": "anger",
        "4": "sadness",
        "5": "disgust",
        "6": "shame",
        "7": "guilt",
    }
    isear_emotions = path.join(folder, "isear.csv")
    with open(isear_emotions) as f:
        next(f)
        for line in f:
            fields = line.split("|", maxsplit=40)
            emotion = fields[11]
            text, _, __ = fields[-1].rsplit("|", maxsplit=2)
            emoname = mapping.get(emotion)
            d = emotion_mapping({emoname: 1}, mapping.values())
            yield {
                "source": "isear",
                "text": text.strip().replace(" \u00e1 ", " "),
                "emotions": d,
                "split": None,
            }


def extract_tales_emotion(folder):
    # ie "2" stands for angry+disgust
    # we unmerge these, by looking into .emmod files
    mapping = {
        "A": "anger",
        "F": "fear",
        "H": "joy",
        "Sa": "sadness",
        "Su": "surprise",
        "D": "disgust",
        "N": "noemo",
        # "Su+": "su+",
        # "Su-": "su-",
        "Su+": "surprise",
        "Su-": "surprise",
    }

    files = itertools.chain.from_iterable(
        iglob(path.join(folder, author, "emmood", "*.emmood"))
        for author in ("Potter", "HCAndersen", "Grimms")
    )
    for fname in files:
        with open(fname) as f:
            for line in f:
                _, emolabel, moodlabel, text = line.split("\t")
                a, b = emolabel.split(":")
                c, d = moodlabel.split(":")
                # force emotion agreement
                if a != b:  # or c != d: (this would force mood agreement, too)
                    l = [a, b]
                    if "N" in l and len(set(l)) != 1:
                        l.remove("N")
                        a = l[0]
                    else:
                        continue
                # fix this, you need two classes for +Su and -Su
                # for some reason I get only Su-
                emotion = a
                emoname = mapping.get(emotion)
                if emoname is None:
                    continue
                d = emotion_mapping({emoname: 1}, mapping.values())
                yield {
                    "source": "tales-emotion",
                    "text": text.strip(),
                    "emotions": d,
                    "optional": {"_tales_source": fname.split("/")[-1]},
                }


def extract_emotiondata_aman(folder):
    # also took the annotations with high agreement
    subfolder = "Emotion-Data/Benchmark"
    mapping = {
        "hp": "joy",
        "ne": "noemo",
        "dg": "disgust",
        "sd": "sadness",
        "sp": "surprise",
        "fr": "fear",
        "ag": "anger",
    }

    benchmark_gold = path.join(folder, subfolder, "category_gold_std.txt")
    with open(benchmark_gold) as f:
        for line in f:
            emotion, _, text = line.split(" ", maxsplit=2)
            emoname = mapping.get(emotion)
            d = emotion_mapping({emoname: 1}, mapping.values())
            yield {
                "source": "emotiondata-aman",
                "text": text.strip(),
                "emotions": d,
                "split": None,
            }


def extract_emotion_cause(folder):
    subfolder = "Dataset"
    mapping = {
        "anger": "anger",
        "happy": "joy",
        "sad": "sadness",
        "surprise": "surprise",
        "fear": "fear",
        "disgust": "disgust",
    }
    nocause = path.join(folder, subfolder, "Emotion Cause.txt")
    cause = path.join(folder, subfolder, "No Cause.txt")

    emotion_pattern = re.compile(r"^<([^>]+)>")
    tag_pattern = re.compile(r"<[^>]+?>")
    for fname in (cause, nocause):
        with open(fname) as f:
            for line in f:
                emotion = emotion_pattern.findall(line)[0]
                emoname = mapping.get(emotion)
                d = emotion_mapping({emoname: 1}, mapping.values())
                yield {
                    "source": "emotion-cause",
                    "text": tag_pattern.sub("", line.strip()),
                    "emotions": d,
                    "split": None,
                }


def extract_emo_bank(folder):
    with open(folder + "/corpus/emobank.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row["text"]
            valence = float(row["V"])
            arousal = float(row["A"])
            dominance = float(row["D"])
            yield {
                "source": "emobank",
                "text": text,
                "emotions": emotion_mapping({}, []),
                "VAD": {
                    "valence": valence,
                    "arousal": arousal,
                    "dominance": dominance,
                },
                "split": None,
            }


def extract_affectivetext(folder):
    tag_pattern = re.compile(r"<[^>]+?>")
    # need to change the columns?
    columns = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
    for part in ("trial", "test"):
        # TODO add a field _train, _test , _dev in the unified dataset
        subfolder = f"AffectiveText.{part}"
        textfile = path.join(folder, subfolder, f"affectivetext_{part}.xml")
        emofile = path.join(
            folder, subfolder, f"affectivetext_{part}.emotions.gold"
        )
        with open(textfile) as t, open(emofile) as e:
            next(t)  # skip header
            for tline, eline in zip(t, e):
                _, *emotions = eline.strip().split(" ")
                # emoname = mapping.get(emotion)
                # d = emotion_mapping({emoname: 1}, mapping.values())
                d = {
                    columns[i]: int(emo) / 100 for i, emo in enumerate(emotions)
                }
                d = emotion_mapping(d, columns)

                yield {
                    "source": "affectivetext",
                    "text": tag_pattern.sub("", tline).strip(),
                    "emotions": d,
                    "split": part,
                }


def extract_dailydialogs(folder):
    subfolder = "ijcnlp_dailydialog"
    mapping = dict(
        zip(
            range(7),
            ["noemo", "anger", "disgust", "fear", "joy", "sadness", "surprise"],
        )
    )
    fname = path.join(folder, subfolder, "dialogues_emotion.txt")
    gname = path.join(folder, subfolder, "dialogues_text.txt")
    with open(fname) as f, open(gname) as g:
        for fline, gline in zip(f, g):
            for emoval, text in zip(
                fline.strip().split(" "), gline.split("__eou__")
            ):
                emoname = mapping[int(emoval)]
                d = emotion_mapping({emoname: 1}, mapping.values())
                yield {
                    "source": "dailydialog",
                    "text": text.strip(),
                    "emotions": d,
                }


def extract_crowdflower(folder):
    mapping = {
        "anger": "anger",
        "enthusiasm": "joy",
        "fun": "joy",
        "happiness": "joy",
        "hate": "anger",
        "neutral": "noemo",
        "sadness": "sadness",
        "surprise": "surprise",
        "worry": "fear",
        "love": "love",
        "boredom": "disgust",
        "relief": "joy",
        "empty": "noemo",
    }

    with open(path.join(folder, "text_emotion.csv")) as f:
        reader = csv.DictReader(f)
        for line in reader:
            emoname = mapping.get(line["sentiment"])
            d = emotion_mapping({emoname: 1}, mapping.values())
            text = line["content"].encode("latin1").decode("utf8")
            # Skip tweets with messed-up encoding, like:
            # "vï¿½o banh ch?y lï¿½ng vï¿½ng trong phï¿½ng"
            # These "ï¿½" sequences are badly encoded U+FFFDs (unicode
            # replacement characters), meaning we've lost information
            # This could be fixed by re-crawling those tweets instead of
            # skipping them.
            if "\ufffd" in text:
                continue
            yield {
                "source": "crowdflower",
                "text": text,
                "emotions": d,
                "split": None,
            }


def extract_meld(sub_dataset):
    """ Extract all data in MELD """

    def inner(folder):
        mapping = {
            "anger": "anger",
            "joy": "joy",
            "neutral": "noemo",
            "sadness": "sadness",
            "surprise": "surprise",
            "fear": "fear",
            "disgust": "disgust",
            "Joyful": "joy",
            "Sad": "sadness",
            "Neutral": "noemo",
            "Scared": "fear",
            "Mad": "anger",
        }

        for filename in iglob(f"{folder}/*"):
            part = (
                "train"
                if "train" in filename
                else "test"
                if "test" in filename
                else "dev"
            )
            with open(filename, "rb") as f:
                f = WeirdlyEncodedFile(f)
                reader = csv.DictReader(f)
                for line in reader:
                    emoname = mapping.get(line["Emotion"])
                    if not emoname:
                        continue
                    d = emotion_mapping({emoname: 1}, mapping.values())
                    text = line["Utterance"]
                    yield {
                        "source": sub_dataset,
                        "text": text,
                        "emotions": d,
                        "split": part,
                    }

    return inner


def extract_ssec(folder):
    mappings = {
        "anger": 0,
        "trust": 1,
        "disgust": 2,
        "fear": 3,
        "joy": 4,
        "sadness": 5,
        "surprise": 6,
    }

    def judge(ls):
        c = Counter(ls)
        # Do we have only one answer?
        if len(c) == 1:
            return int(ls[0])
        # How many people disagree with the majority?
        disagreers = c.most_common()[1][1]
        if disagreers <= len(ls) // 2 - 1:
            return int(c.most_common()[0][0])
        else:
            return None

    def handle_line(line):
        csv_part, *fields = line.split("\t")
        reader = csv.reader([csv_part])
        tweet = next(reader)[0]
        aggregates = [[x for x in fields[i::8] if x != -1] for i in range(8)]
        # No annotation
        if "XXXXXXXXXXXX" in line:
            return
        if len(aggregates[0]) < 2:  # less than two annotators
            return
        d = {
            "anger": 0,
            "joy": 0,
            "sadness": 0,
            "disgust": 0,
            "fear": 0,
            "surprise": 0,
        }
        for emotion in mappings:
            judgements = aggregates[mappings[emotion]]
            # we take a 1 if at least 1 annotator annotated it with
            # no more judge(judgments)

            if "1" in judgements:
                verdict = 1
            else:
                verdict = 0
            if verdict is None:
                break
            d[emotion] = verdict
        else:  # no-break
            yield {
                "source": "ssec",
                "text": tweet,
                "emotions": emotion_mapping(d, d.keys()),
                "split": part,
            }

    for part in ("test", "train"):
        with open(
            path.join(folder, f"emotioncorpus-{part}.csv"), encoding="latin1"
        ) as f:
            for line in f:
                yield from handle_line(line)


def extract_fb_va(folder):
    with open(folder + "/dataset-fb-valence-arousal-anon.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row["Anonymized Message"]
            arousal = (int(row["Arousal1"]) + int(row["Arousal2"])) / 2
            valence = (int(row["Valence1"]) + int(row["Valence2"])) / 2
            yield {
                "source": "fb-valence-arousal-anon",
                "text": text,
                "emotions": emotion_mapping({}, []),
                "VAD": {
                    "valence": valence,
                    "arousal": arousal,
                    "dominance": None,
                },
                "split": None,
            }


def extract_EGK(folder):
    with open(folder + "/fanfic_test.jsonl") as f:
        for line in f:
            row = json.loads(line)
            emotions = row["emotions"]
            emotions["noemo"] = emotions.pop("no-emo")
            yield {
                "source": row["source"],
                "text": row["text"],
                "emotions": emotion_mapping(emotions, emotions.keys()),
                "VAD": {"valence": None, "arousal": None, "dominance": None},
                "split": None,
            }


if __name__ == "__main__":
    extractors = {
        "EmoBank": extract_emo_bank,
        "fb-valence-arousal-anon": extract_fb_va,
        "crowdflower": extract_crowdflower,
        "dailydialog": extract_dailydialogs,
        "emotion-cause": extract_emotion_cause,
        "emotiondata-aman": extract_emotiondata_aman,
        "affectivetext": extract_affectivetext,
        "isear": extract_isear,
        "tales-emotion": extract_tales_emotion,
        "grounded_emotions": extract_grounded_emotions,
        "ssec": extract_ssec,
        "TEC": extract_tec,
        "emoint": extract_emoint,
        "electoraltweets": extract_electoraltweets,
        "EGK": extract_EGK,
        "MELD": extract_meld("meld"),
        "MELD_Dyadic": extract_meld("meld-dya"),
        "emorynlp": extract_meld("emorynlp"),
        "jointMultitaskEmo": extract_jointMultitaskEmo,
        "README.md": None,
    }
    meta_info = {
        "emotion_model": {
            "Ekman": [
                "dailydialog",
                "emotion_cause",
                "tales-emotion",
                "affectivetext",
                "TEC",
                "MELD",
                "MELD_Dyadic",
                "emorynlp",
            ],
            "VA": ["fb-valence-arousal-anon"],
            "Plutchik": ["ssec", "EGK", "jointMultitaskEmo"],
            "Ekman+ne": ["emotiondata-aman"],
            "VAD": ["EmoBank"],  #
            "Ekman-disgust-surprise": ["emoint"],
            "Ekman+CF": ["crowdflower"],
            "Ekman+ET": ["electoraltweets"],
            "HappySad": ["grounded_emotions"],
            # read the paper and table 1
        },
        "annotation_procedure": {
            "crowdsourcing": ["crowdflower"],
            "expert annotation": ["emoint", "TEC", "EGK"],
        },
        "domain": {
            "tweets": [
                "TEC",
                "ssec",
                "electoraltweets",
                "emoint",
                "crowdflower",
                "grounded_emotions",
            ],
            "facebook-messages": ["fb-valence-arousal-anon"],
            "headlines": ["affectivetext", "EmoBank"],
            "conversations": ["dailydialog", "MELD", "MELD_Dyadic", "emorynlp"],
            "blogposts": ["emotiondata-aman"],
            "emotional_events": ["isear"],
            "artificial_sentences": ["emotion-cause"],
            "fairytale_sentences": ["tales-emotion"],
            "fanfiction": ["EGK"],
            "multidomain": ["jointMultitaskEmo"],
        },
        "labeled": {
            "multi": [
                "affectivetext",
                "ssec",
                "fb-valence-arousal-anon",
                "EGK",
            ],
            "single": [
                "TEC",
                "electoraltweets",
                "emoint",
                "crowdflower",
                "grounded_emotions",
                "dailydialog",
                "emotiondata-aman",
                "isear",
                "emotion-cause",
                "tales-emotion",
                "MELD",
                "MELD_Dyadic",
                "emorynlp",
                "jointMultitaskEmo",
            ],
        },
    }
    metadata = {}
    for key in meta_info:
        for value in meta_info[key]:
            for dataset in meta_info[key][value]:
                metadata.setdefault(dataset, {})[key] = value

    counter = itertools.count()
    with open("unified-dataset.jsonl", "w") as f:
        for folder in itertools.chain(
            iglob("datasets/*"), iglob("own-datasets/*")
        ):
            name = folder.split("/")[-1]
            if name not in extractors:
                print("No extractor defined for", name)
                continue
            elif extractors[name] is None:
                continue
            print("Extracting from", name)
            for line in extractors[name](folder):
                d = {
                    "id": next(counter),
                    "VAD": {
                        "valence": None,
                        "arousal": None,
                        "dominance": None,
                    },
                }
                d.update(line)
                d.update(meta_info.get(name, metadata[name]))
                json.dump(d, f)
                f.write("\n")
    print("All done; Created unified-dataset.jsonl")
