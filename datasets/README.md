# WIP

# Emotion Datasets Survey README

This file is a meta-README for all the datasets used in emotion recognition that are redistributatble or obtainable. 


## Datasets overview

**Summary Table**

|Dataset | Author | Year | License | Description | Format | Size | Emotion categories |
|:------:| :----: | :--: | :-----: | :---------: | :----: | :--: | :---------:|
|affectivetext|Strapparava & Mihalcea|2007||Classification of emotions in news headlines|SGML/txt|250 headlines|anger, disgust, fear, joy, sadnees, surprise, V|
|crowdflower_data|CrowdFlower|2016|available to download|Annotated dataset of tweets via crowdsourcing.|csv|40k tweets| anger, enthusiasm, fun, happiness, hate, neutral, sadness, surprise, worry, love, boredom, worry, relief, empty|
|dailydialog|Li Yanrand et al|2017|available to download|Manually labelled conversations dataset with topics and emotions|text|13k dialogs|anger, disgust, fear, joy, sadness, surprise|
|emotion-cause|Diman Ghazi&Diana Inkpen&Stan Szpakowicz|2015| research only |Automatically built dataset annotated with emotion and the stimulus using FrameNet’s emotions-directed frame|XML|820 sents + 1594 sents|anger, sad, happy, surprise, fear, disgust|
|EmoBank|Sven Buechel|2017|redistributable CC-BY 4.0|Large-scale corpus annotated with emotion according to  VAD scheme|text|10k| VAD
|emotiondata-aman|Saima Aman&Stan Szpakowicz|2007|obtainable upon request|Manually annotated corpus with emotion categories. The agreement on emotion categories was ~0.66. |text|~15k sents|joy, neutral, disgust, sadness, surprise, fear, anger|
|fb-valence-arousal-annon|Preotiuc Pietro|2016|available to download| description |csv|2.8k posts|VA|
|grounded_emotions|Liu, V.&Banea, C.&Mihalcea|2017|available to download|They look into wheter the effect of weather, news events, relates to the tweet sentiment |text|2.5k tweets|joy, sadness|
|isear| Klaus R. Scherer and Harald Wallbott|1990|available to download|reported situations in which emotions were experienced|text (mdb/sav)|3000 docs| joy, fear, anger, sadness, disgust,shame, guilt|
|tales-emotions|Cecilia Ovesdotter Alm|2005|gplv3| Dataset of manually annotated tales used in a document classification task |text|15k sents| angry, disgusted, fearful, happy, sad, surprised, mood (positive, negative)|
|emoint|
|electoraltweets|



## Datasets

### Affective Text (Test Corpus of SemEval 2007)

* Author: Carlo Strapparava & Rada Mihalcea
* Description: Classification of emotions in news headlines (Semeval 2007)
* Size: 250 headlines
* Categories: 0: anger, 1: disgust, 2: fear, 3: joy, 4: happiness, 5: sadness, 6: surprise
* Annotation: emotions (anger disgust fear joy sadness surprise) + valence
* Approaches:
* Format: SGML+text
* Link: http://web.eecs.umich.edu/~mihalcea/downloads/AffectiveText.Semeval.2007.tar.gz

#### Sample:

- emotions (id anger disgust fear joy sadness surprise):
    `245 0 0 0 54 0 6`
- valence (id valence):
    `245 52`
- corpus:
    `<instance id="245">Melua's deep sea gig sets record </instance>`

#### Papers:

* [Strapparava, C., & Mihalcea, R. (2007, June). Semeval-2007 task 14: Affective text. In Proceedings of the 4th International Workshop on Semantic Evaluations (pp. 70-74). Association for Computational Linguistics.](http://delivery.acm.org/10.1145/1630000/1621487/p70-strapparava.pdf)

#### Bibtex entries for the papers:

    @inproceedings{strapparava2007semeval,
        title={Semeval-2007 task 14: Affective text},
        author={Strapparava, Carlo and Mihalcea, Rada},
        booktitle={Proceedings of the 4th International Workshop on Semantic Evaluations},
        pages={70--74},
        year={2007},
        organization={Association for Computational Linguistics}
    }


### Crowdflower (The Emotion in Text dataset by CrowdFlower)

* Author: CrowdFlower
* Description: Dataset of tweets labelled with emotion
* Size: 40000 tweets
* Categories: empty, sadness, enthusiasm, neutral, worry, sadness, love, fun, hate, happiness, relief, boredom, surprise, anger
* Annotation: emotions in tweets ("tweet_id","sentiment","author","content")
* Approaches:
* Format: csv

* Link: http://www.crowdflower.com/wp-content/uploads/2016/07/text_emotion.csv
#### Sample:

- corpus:
`1956967341,"empty","xoshayzers","@tiffanylue i know  i was listenin to bad habit earlier and i started freakin at his part =["`

#### Papers:

*

#### Bibtex entries for the papers:

?
#### Remarks:

- I don't know how the annotations were made, they don't seem ok. Also the users should be annonymized.
- many bad annotations

### DailyDialog

* Authors: Li, Yanran and Su, Hui and Shen, Xiaoyu and Li, Wenjie and Cao, Ziqiang and Niu, Shuzi
* Description: A manually labelled conversations dataset
* Size: 13118 sentences
* Categories: 0: no emotion, 1: anger, 2: disgust, 3:fear, 4: happiness, 5: sadness, 6: surprise
* Annotation: topic number {1..10} (where 4 stands for Attitude&Emotion) + act number {1..4} + emotion number {0..6}
* Approaches:
* Format: text
* Link: http://yanran.li/dailydialog.html

#### Sample:

- corpus: dialogues_text.txt
    `The kitchen stinks . __eou__ I'll throw out the garbage . __eou__`
- topic: dialogues_topic.txt
    `1`
- act: dialogues_act.txt
    `3 4`
- emotion: dialogues_emotion.txt
    `2 0`

#### Papers:

* [Li, Y., Su, H., Shen, X., Li, W., Cao, Z., & Niu, S. (201A7). DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset. arXiv preprint arXiv:1710.03957.](https://arxiv.org/pdf/1710.03957)

#### Bibtex entries for the papers:

    @article{li2017dailydialog,
        title={DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset},
        author={Li, Yanran and Su, Hui and Shen, Xiaoyu and Li, Wenjie and Cao, Ziqiang and Niu, Shuzi},
        journal={arXiv preprint arXiv:1710.03957},
        year={2017}
    }


### Emotion-stimulus data

* Authors: Diman Ghazi, Diana Inkpen & Stan Szpakowicz
* Description: They automatically build a dataset annotated with both the emotion and the stimulus using FrameNet’s emotions-directed frame
* Size: 820 sentences with both cause and emotion and 1594 sentences marked with their emotion tag 
* Categories: happiness, sadness, anger, fear, surprise, disgust, shame
* Annotation: emotion + stimulus
* Approaches: CRF learner, a sequential learning model to detect the emotion stimulus spans in emotion-bearing sentences
* Format: "XML" (with incorrect closing tags)
* Link: http://www.site.uottawa.ca/~diana/resources/emotion_stimulus_data

#### Sample:

- emotions and cause:
    `<happy>These days he is quite happy <cause>travelling by trolley<\cause> . <\happy>`
- emotions only:
    `<anger>Bernice was so angry she could hardly speak . <\anger>`

#### Papers:

* [Ghazi, D., Inkpen, D., & Szpakowicz, S. (2016, April). Detecting Emotion Stimuli in Emotion-Bearing Sentences. In CICLing (2) (pp. 152-165).](http://www.site.uottawa.ca/~diana/publications/90420152.pdf)

#### Bibtex entries for the papers:

    @inproceedings{ghazi2016detecting,
        title={Detecting Emotion Stimuli in Emotion-Bearing Sentences.},
        author={Ghazi, Diman and Inkpen, Diana and Szpakowicz, Stan},
        booktitle={CICLing (2)},
        pages={152--165},
        year={2015}
    }

#### Remarks:
 stimulus = cause


### Emotion data Aman

* Authors: Aman Saima
* Description: Identifying emotions in text
* Size: 15205 sentences (173 blogposts)
* Categories: happiness, sadness, anger, disgust, surprise, fear, ne (no emotion)
* Annotation: emotion category label + emotion intensity label + emotion idicators
* Approaches:
* Link: ?
* Terms/License: Available upon request

#### Format
The files AnnotSet1.txt and AnnotSet2.txt are annotated one sentence per line, with:

    <id> <emotion category label> <emotion intensity label> <emotion indicators>

- _id_ is a number followed by a dot.
- _emotion category label_ is one of: `hp` (happiness), `sd` (sadness), `ag` (anger), `dg` (disgust), `sp` (surprise), `fr` (fear), or `ne` (no emotion). If the emotion category label is "ne" (no emotion), the line ends here.
- _emotion intensity label_ is one of: `h` (high), `l` (low), or `m` (medium)
- _emotion indicators_ are the words that were found by annotators to be emotionally charged. They are sometimes (but not always) seperated by commas.

basefile.txt contains the corresponding sentences, one sentence per line. Each sentence is preceeded by its number, a dot, and a space.

blogdata.txt is an XML file.

The file category_gold_std.txt contains all those sentences for which the annotators agreed on the emotion category and has the following format:

    <emotion category label> <sentence number> <sentence>

Similarly intensity_gold_std.txt contains all those sentences for which agreement on emotion intensity was met and has the following format:

    <emotion intensity label> <sentence number> <sentence>

#### Sample:
first set (Annotator A) AnnotSet1.txt:
`43. hp m played, fun, baby, games`

second set (Annotators B,C and D) AnnotSet2.txt:
`43. hp m fun baby games`

basefile.txt
`43. We played fun baby games and caught up on some old times.`

unannotated data: blogdata.txt

    <block id="block 3">
      <!-- ... -->
      <resource id="3.43">We played fun baby games and caught up on some old times.</resource>
      <!-- ... -->
    </block>

category_gold_std.txt:
`p 43. We played fun baby games and caught up on some old times.`

intensity_gold_std.txt:
`m 43. We played fun baby games and caught up on some old times.`

#### Papers:

*[Aman, S. (2007). Recognizing emotions in text (Doctoral dissertation, University of Ottawa (Canada)).](https://ruor.uottawa.ca/bitstream/10393/27501/1/MR34054.PDF)
*[Aman, S., & Szpakowicz, S. (2007). Identifying expressions of emotion in text. In Text, speech and dialogue (pp. 196-205). Springer Berlin/Heidelberg.](http://ccc.inaoep.mx/~villasen/bib/Identifying%20Expression%20of%20Emotion%20in%20Text.pdf)

#### Bibtex entries for the papers:

    @phdthesis{aman2007recognizing,
        title={Recognizing emotions in text},
        author={Aman, Saima},
        year={2007},
        school={University of Ottawa (Canada)}
    }

    @inproceedings{aman2007identifying,
        title={Identifying expressions of emotion in text},
        author={Aman, Saima and Szpakowicz, Stan},
        booktitle={Text, speech and dialogue},
        pages={196--205},
        year={2007},
        organization={Springer}
    }

#### Remarks:


### EmoBank

* Authors: Sven Buechel
* Description: A large-scale text corpus manually annotated with emotion according to the psychological Valence-Arousal-Dominance scheme.
* Size: 10k sentences balancing multiple genres
* Categories:
* Annotation: Each sentence was annotated according to both the emotion which is expressed by the writer, and the emotion which is perceived by the readers.
* Approaches:
* Link: https://github.com/JULIELab/EmoBank
* Terms/License: 

#### Format:

The folder `corpus` contains four tsv files and one script in R.
The file  raw.tsv contains the textual data of this corpus (id	sentence)
The file  meta.tsv contains genre-information (id	document	category	subcategory)
The files reader.tsv and writer.tsv contain the emotion meta-data for the writer and reader perspective
(id	Arousal	Dominance	Valence	sd.Arousal	sd.Dominance	sd.Valence	freq)

These meta-data files contain columns for the three emotional dimensions (Valence, Arousal and Dominance) as well as the standard deviation of the human ratings for the individual item. The last column, "freq", gives the number of valid ratings for this item (after the filtering process described in the paper).

The IDs for sentences taken from MASC follow the pattern \<documentName>\_\<beginIndex>\_\<endIndex> which refers to the file names and the sentence boundary annotations as included in the MASC release 3.0.0. For the news headlines taken from SemEval, the IDs follow the pattern SemEval\_\<originalSemEvalId>.

The script combine.R combines those individual files into a single R data frame

#### Sample:
raw.tsv:
`Acephalous-Cant-believe_4_47	I can't believe I wrote all that last year.`

meta.tsv:
`Acephalous-Cant-believe_4_47	Acephalous-Cant-believe	blog	 .`

reader.tsv:
`110CYL068_1036_1079	3.2	3	3	0.4	0	0	5`

writer.tsv:
`110CYL068_1036_1079	2.8	3.4	3	0.979795897113271	0.489897948556636	0	5`

combine.R

#### Papers:

*[Buechel, S., & Hahn, U. (2017). EMOBANK: Studying the Impact of Annotation Perspective and Representation Format on Dimensional Emotion Analysis. EACL 2017, 578.](https://www.aclweb.org/anthology/E/E17/E17-2.pdf#page=610)
*[Buechel, S., & Hahn, U. (2017). Readers vs. Writers vs. Texts: Coping with Different Perspectives of Text Understanding in Emotion Annotation. LAW XI 2017, 1.](http://www.aclweb.org/anthology/W/W17/W17-08.pdf#page=13)

#### Bibtex entries for the papers:

    @article{buechel2017emobank,
        title={EMOBANK: Studying the Impact of Annotation Perspective and Representation Format on Dimensional Emotion Analysis},
        author={Buechel, Sven and Hahn, Udo},
        journal={EACL 2017},
        pages={578},
        year={2017}
    }

    @article{buechel2017readers,
        title={Readers vs. Writers vs. Texts: Coping with Different Perspectives of Text Understanding in Emotion Annotation},
        author={Buechel, Sven and Hahn, Udo},
        journal={LAW XI 2017},
        pages={1},
        year={2017}
    }

#### Remarks:

* A subset of the corpus have been previously annotated according to Ekmans 6 Basic Emotions (Strapparava and Mihalcea, 2007) so that mappings between both representation formats is possible.


### The valence and arousal in facebook posts

* Authors: Preotiuc-Pietro  et al
* Description: The dataset is used to train prediction models for each two dimensions of valence and arousals
* Size: 2896 facebook posts
* Annotation: Valence + Arousal
* Approaches:
* Format: text/csv
* Link: http://wwbp.org/downloads/public_data/dataset-fb-valence-arousal-anon.csv

#### Format:
The dataset is only one csv file named: dataset-fb-valence-arousal-annon.csv. The header of the csv file is:

`Anonymized Message,Valence1,Valence2,Arousal1,Arousal2`

where:
`Arousal` is intensity of the affective content, rated on a nine point scale from 1 (neutral/objective post) to 9 (very high) and `Valence` represents the polarity of the affective content in a post, rated on a nine point scale from 1 (very negative) to 5 (neutral/objective) to 9 (very positive)

#### Sample:
`"Ergh, Anatomy, 16 page outline AND 15 pages of pictures to label by TOMORROW, I curse you procrastination. May you rot forever in purgatory",3,3,4,5`

#### Papers:

*[Preoţiuc-Pietro, D., Schwartz, H. A., Park, G., Eichstaedt, J., Kern, M., Ungar, L., & Shulman, E. (2016). Modelling valence and arousal in facebook posts. In Proceedings of the 7th Workshop on Computational Approaches to Subjectivity, Sentiment and Social Media Analysis (pp. 9-15).](http://www.aclweb.org/anthology/W16-0404)

#### Bibtex entries for the papers:

    @inproceedings{preoctiuc2016modelling,
        title={Modelling valence and arousal in facebook posts},
        author={Preo{\c{t}}iuc-Pietro, Daniel and Schwartz, H Andrew and Park, Gregory and Eichstaedt, Johannes and Kern, Margaret and Ungar, Lyle and Shulman, Elisabeth},
        booktitle={Proceedings of the 7th Workshop on Computational Approaches to Subjectivity, Sentiment and Social Media Analysis},
        pages={9--15},
        year={2016}
    }

#### Remarks:
?

### Grounded Emotions

* Authors: Vicki Liu&Carmen Banea&Rada Mihalcea
* Description: They look into wheter the effect of weather, news events, relates to the tweet sentiment
* Size: 2.5k tweets, ~0.5k news
* Version: 1.0
* Categories: only happiness and sadness
* Annotation: tweets
* Approaches:
* Format:
* Link: http://web.eecs.umich.edu/~mihalcea/downloads.html#GroundedEmotions
* Contact: vickliu@umich.edu

#### Format:
Data is separated into collected data and derived data.

#### Sample:


#### Papers:

*[Liu, V., Banea, C., & Mihalcea, R. Grounded Emotions. Chicago](http://web.eecs.umich.edu/~mihalcea/papers/liu.acii17.pdf)


#### Bibtex entries for the papers:

    @inproceedings{Liu17,
        author = {Liu, V. and C. Banea and R. Mihalcea},
        title={Grounded Emotions},
        booktitle = {International Conference on Affective Computing and Intelligent Interaction (ACII 2017)},
        address = {San Antonio, Texas},
        year = {2007}
    }

#### Remarks:
- this dataset comes with a very detailed README file.
- more about sentiment than emotion, therefore not sure of how interesting this is for us.
- actually this could be extended for all emotions. why didn't they include all emotions?

### ISEAR (International Survey On Emotion Antecedents And Reactions)
* Authors: Klaus R. Scherer and Harald Wallbott
* Description: Over a period of many years during the 1990s, a large group of psychologists all over the world collected data in the ISEAR project, directed by Klaus R. Scherer and Harald Wallbott. Student respondents, both psychologists and non-psychologists, were asked to report situations in which they had experienced all of 7 major emotions (joy, fear, anger, sadness, disgust, shame, and guilt). In each case, the questions covered the way they had appraised the situation and how they reacted. The final data set thus contained reports on seven emotions each by close to 3000 respondents in 37 countries on all 5 continents.
* Size: 3000 respondents
* Categories: joy, fear, anger, sadness, disgust, shame, guilt
* Annotation:
* Approaches:
* Format: mdb/sav/csv
* Links: http://emotion-research.net/toolbox/toolboxdatabase.2006-10-13.2581092615
        http://www.affective-sciences.org/index.php/download_file/view/395/296/
        
#### Sample:


#### Papers:

*

#### Bibtex entries for the papers:


#### Remarks:

### tales-emotion

* Authors: Cecilia Ovesdotter Alm
* Description: document classification task --classify the emotional affinity of sentences in the narrative domain of children's fairy tales
* Size: 185 fairytales, 15292 sents
* Categories: 
* Annotation: emotions (Angry, Disgusted, Fearful, Happy, Sad) + moods (Positively surprised / Negatively surprised) + Neutral
* Approaches: supervised machine learning with the SNoW learning architecture
* Link: http://people.rc.rit.edu/~coagla/affectdata/index.html
* Contact: ebbaalm@uiuc.edu

#### Format:

The document tales.txt lists the basenames of the story files

The files ending in `.emmod` have the following format:
`SentID:SentID	1emLabelA:1emLabelB	MoodLabelA:MoodLabelB	Sent`

They contain sentences with unmerged labels for the the annotators (A and B).
The labels are: `A`: Angry, `D`: Disgusted, `F`: Fearful, `H`: Happy, `N`: Neutral, `Sa`: Sad for Primary emotion (1em) or `S`: Sad when is Mood, `Su+`: Positively Surprised for 1em, or `+` for Mood, `Su-` for 1em, or `-` for Mood.

The files ending in `.sent.okpuncs` contain the sentences and have the following format:
`sent		`

The files ending in ` .sent.okpuncs.props.pos` contain the sentences pos tagged and have the following format:
`(Tag word):(Tag word) [...]`

The files ending in `.agreeID` contain the IDs (numbers)  of the sentences with full agreement on the affect. The labels used here are: `Angry-Disgusted` (merged), `Fearful`, `Happy`, `Sad`, and `Surprised` (merged). Neutral labels are not in!

The files ending in `.agree` contains only sentences with AFFECTIVE HIGH AGGREMENTS (see description for corresponding agreeID directory). The label codes are: `2`=`Angry-Disgusted`, `3`=`Fearful`, `4`=`Happy`, `6`=`Sad`, `7`=`Surprised` and have the following format:
`SentID@AffectiveLabelCode@Sentence`

#### Sample:

`*.emmood`:
`0:0     N:N     N:N     Once upon a time there was a village shop.`

`*.sent.okpuncs`:
`Once upon a time there was a village shop.`

`*.sent.okpuncs.props.pos`:
`(RB Once):(IN upon):(DT a):(NN time):(EX there):(AUX was):(DT a):(NN village):(NN shop):(. .)`

`*.agreeID`:
`35`

`*.agree`:
`35@3@"It is very unpleasant, I am afraid of the police," said Pickles.`

#### Papers:

*[Alm, C. O., Roth, D., & Sproat, R. (2005, October). Emotions from text: machine learning for text-based emotion prediction. In Proceedings of the conference on human language technology and empirical methods in natural language processing (pp. 579-586). Association for Computational Linguistics.](http://www.aclweb.org/anthology/H/H05/H05-1.pdf#page=615)
*[Alm, C. O., & Sproat, R. (2005). Perceptions of emotions in expressive storytelling. In Ninth European Conference on Speech Communication and Technology.](https://pdfs.semanticscholar.org/6c91/6e257bd9d44d433b3b5307d68423b84cbf8c.pdf)
*[Alm, E. C. O. (2008). Affect in* text and speech. University of Illinois at Urbana-Champaign.](http://cogcomp.cs.illinois.edu/papers/Alm%20thesis(1).pdf)

#### Bibtex entries for the papers:

    @inproceedings{alm2005emotions,
        title={Emotions from text: machine learning for text-based emotion prediction},
        author={Alm, Cecilia Ovesdotter and Roth, Dan and Sproat, Richard},
        booktitle={Proceedings of the conference on human language technology and empirical methods in natural language processing},
        pages={579--586},
        year={2005},
        organization={Association for Computational Linguistics}
    }

    @inproceedings{alm2005perceptions,
        title={Perceptions of emotions in expressive storytelling},
        author={Alm, Cecilia Ovesdotter and Sproat, Richard},
        booktitle={Ninth European Conference on Speech Communication and Technology},
        year={2005}
    }

    @book{alm2008affect,
        title={Affect in* text and speech},
        author={Alm, Ebba Cecilia Ovesdotter},
        year={2008},
        publisher={University of Illinois at Urbana-Champaign}
    }

#### Remarks:


### SSEC

* Authors:
* Description:t
* Size:
* Categories:
* Annotation:
* Approaches:
* Format:
* Link:

#### Sample:


#### Papers:

*

#### Bibtex entries for the papers:


#### Remarks:


### EmoInt

* Authors:
* Description:
* Size:
* Categories:
* Annotation:
* Approaches:
* Format:
* Link:

#### Sample:


#### Papers:

*

#### Bibtex entries for the papers:


#### Remarks:


### Electoral Tweets

* Authors:
* Description:
* Size:
* Categories:
* Annotation:
* Approaches:
* Format:
* Link:

#### Sample:


#### Papers:

*

#### Bibtex entries for the papers:


#### Remarks:




