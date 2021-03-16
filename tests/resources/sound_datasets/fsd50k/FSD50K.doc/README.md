
# Freesound Dataset 50k (FSD50K)

## Citation

If you use the FSD50K dataset, or part of it, please cite our paper:

>Eduardo Fonseca, Xavier Favory, Jordi Pons, Frederic Font, Xavier Serra. "FSD50K: an Open Dataset of Human-Labeled Sound Events", arXiv 2020.

### Data curators

Eduardo Fonseca, Xavier Favory, Jordi Pons, Mercedes Collado, Ceren Can, Rachit Gupta, Javier Arredondo, Gary Avendano and Sara Fernandez

### Contact

You are welcome to contact Eduardo Fonseca should you have any questions at eduardo.fonseca@upf.edu.

## About FSD50K

Freesound Dataset 50k (or **FSD50K** for short) is an open dataset of human-labeled sound events containing 51,197 <a href="https://freesound.org/">Freesound</a> clips unequally distributed in 200 classes drawn from the <a href="https://research.google.com/audioset/ontology/index.html">AudioSet Ontology</a> [1]. FSD50K has been created at the <a href="https://www.upf.edu/web/mtg">Music Technology Group of Universitat Pompeu Fabra</a>.

What follows is a brief summary of FSD50K's most important characteristics. Please have a look at our paper (especially Section 4) to extend the basic information provided here with relevant details for its usage, as well as discussion, limitations, applications and more.


**Basic characteristics:**

- FSD50K is composed mainly of sound events produced by physical sound sources and production mechanisms.  
- Following AudioSet Ontology’s main families, the FSD50K vocabulary encompasses mainly *Human sounds*, *Sounds of things*, *Animal*, *Natural sounds* and *Music*.
- The dataset has 200 sound classes (144 leaf nodes and 56 intermediate nodes) hierarchically organized with a subset of the AudioSet Ontology. The vocabulary can be inspected in `vocabulary.csv` (see Files section below).
- FSD50K contains 51,197 audio clips totalling 108.3 hours of audio.
- The audio content has been manually labeled by humans following a data labeling process using the <a href="https://annotator.freesound.org/">Freesound Annotator</a> platform [2]. 
- Clips are of variable length from 0.3 to 30s, due to the diversity of the sound classes and the preferences of Freesound users when recording sounds.
- Ground truth labels are provided at the clip-level (i.e., weak labels).
- The dataset poses mainly a multi-label sound event classification problem (but also allows a variety of sound event research tasks, see Sec. 4D).
- All clips are provided as uncompressed PCM 16 bit 44.1 kHz mono audio files.
- The audio clips are grouped into a development (*dev*) set and an evaluation (*eval*) set such that they do not have clips from the same Freesound uploader.

**Dev set:**

- 40,966 audio clips totalling 80.4 hours of audio
- Avg duration/clip: 7.1s
- 114,271 smeared labels (i.e., labels propagated in the upwards direction to the root of the ontology)
- Labels are correct but could be occasionally incomplete
- A train/validation split is provided (Sec. 3H). If a different split is used, it should be specified for reproducibility and fair comparability of results (see Sec. 5C of our paper)  


**Eval set:**

- 10,231 audio clips totalling 27.9 hours of audio
- Avg duration/clip: 9.8s
- 38,596 smeared labels
- Eval set is labeled exhaustively (labels are correct and complete for the considered vocabulary)


**NOTE:** All classes in FSD50K are represented in AudioSet, except `Crash cymbal`, `Human group actions`, `Human voice`, `Respiratory sounds`, and `Domestic sounds, home sounds`.


## License

All audio clips in FSD50K are released under Creative Commons (CC) licenses. Each clip has its own license as defined by the clip uploader in Freesound, some of them requiring attribution to their original authors and some forbidding further commercial reuse. For attribution purposes and to facilitate attribution of these files to third parties, we include a mapping from the audio clips to their corresponding licenses. The licenses are specified in the files `dev_clips_info_FSD50K.json` and `eval_clips_info_FSD50K.json`. These licenses are CC0, CC-BY, CC-BY-NC and CC Sampling+.

In addition, FSD50K as a whole is the result of a curation process and it has an additional license: FSD50K is released under <a href="https://creativecommons.org/licenses/by/4.0/">CC-BY</a>. This license is specified in the `LICENSE-DATASET` file downloaded with the `FSD50K.doc` zip file.

 
## Files

FSD50K can be downloaded as a series of zip files with the following directory structure:

<div class="highlight"><pre><span></span>root
│  
└───FSD50K.dev_audio/                   Audio clips in the dev set
│  
└───FSD50K.eval_audio/                  Audio clips in the eval set
│   
└───FSD50K.ground_truth/                Files for FSD50K's ground truth
│   │   
│   └─── dev.csv            				  Ground truth for the dev set
│   │        
│   └─── eval.csv          				      Ground truth for the eval set            
│   │            
│   └─── vocabulary.csv                       List of 200 sound classes in FSD50K 
│   
└───FSD50K.metadata/                    Files for additional metadata
│   │            
│   └─── class_info_FSD50K.json               Metadata about the sound classes
│   │            
│   └─── dev_clips_info_FSD50K.json           Metadata about the dev clips
│   │            
│   └─── eval_clips_info_FSD50K.json          Metadata about the eval clips
│   │            
│   └─── pp_pnp_ratings_FSD50K.json           PP/PNP ratings    
│   │            
│   └─── collection/                          Files for the *sound collection* format   
│   
└───FSD50K.doc/
    │            
    └───README.md                             The dataset description file that you are reading
    │            
    └───LICENSE-DATASET                       License of the FSD50K dataset as an entity   
</pre></div>


Each row (i.e. audio clip) of `dev.csv`  contains the following information:

- `fname`: the file name without the `.wav` extension, e.g., the fname `64760` corresponds to the file `64760.wav` in disk. This number is the Freesound id. We always use Freesound ids as filenames.
- `labels`: the class labels (i.e., the ground truth). Note these class labels are *smeared*, i.e., the labels have been propagated in the upwards direction to the root of the ontology. More details about the label smearing process can be found in Appendix D of our paper. 
- `mids`: the Freebase identifiers corresponding to the class labels, as defined in the <a href="https://github.com/audioset/ontology/blob/master/ontology.json">AudioSet Ontology specification</a>
- `split`: whether the clip belongs to *train* or *val* (see paper for details on the proposed split)

Rows in `eval.csv` follow the same format, except that there is no `split` column.

**NOTE:**  We use a slightly different format than AudioSet for the naming of class labels in order to avoid potential problems with spaces, commas, etc. Example: we use `Accelerating_and_revving_and_vroom` instead of the original `Accelerating, revving, vroom`. You can go back to the original AudioSet naming using the information provided in `vocabulary.csv` (class label and mid for the 200 classes of FSD50K) and the <a href="https://github.com/audioset/ontology/blob/master/ontology.json">AudioSet Ontology specification</a>.



### Files with additional metadata (FSD50K.metadata/)

To allow a variety of analysis and approaches with FSD50K, we provide the following metadata:

 1. `class_info_FSD50K.json`: python dictionary where each entry corresponds to one sound class and contains: `FAQs` utilized during the annotation of the class, `examples` (representative audio clips), and `verification_examples` (audio clips presented to raters during annotation as a quality control mechanism). Audio clips are described by the Freesound id.
 **NOTE:** It may be that some of these examples are not included in the FSD50K release.
 
 2. `dev_clips_info_FSD50K.json`: python dictionary where each entry corresponds to one dev clip and contains: title, description, tags, clip license, and the uploader name. All these metadata are provided by the uploader.

 3. `eval_clips_info_FSD50K.json`: same as before, but with eval clips.
 
 4. `pp_pnp_ratings.json`: python dictionary where each entry corresponds to one clip in the dataset and contains the PP/PNP ratings for the labels associated with the clip. More specifically, these ratings are gathered for the labels validated in **the validation task** (Sec. 3 of paper). This file includes 59,485 labels for the 51,197 clips in FSD50K. Out of these labels:

  - 56,095 labels have inter-annotator agreement (PP twice, or PNP twice). Each of these combinations can be occasionally accompanied by other (non-positive) ratings.  
  - 3390 labels feature other rating configurations such as *i)* only one PP rating and one PNP rating (and nothing else). This can be considered inter-annotator agreement at the ``Present” level; *ii)* only one PP rating (and nothing else); *iii)* only one PNP rating (and nothing else).

  Ratings' legend: PP=1; PNP=0.5; U=0; NP=-1.

  **NOTE:** The PP/PNP ratings have been provided in the *validation* task. Subsequently, a subset of these  clips corresponding to the eval set was exhaustively labeled in the *refinement* task, hence receiving additional labels in many cases. For these eval clips, you might want to check their labels in `eval.csv` in order to have more info about their audio content (see Sec. 3 for details).
 
 5. `collection/`: This folder contains metadata for what we call the ***sound collection format***. This format consists of the raw annotations gathered, featuring all generated class labels without any restriction. 

 We provide the *collection* format to make available some annotations that do not appear in the FSD50K *ground truth* release. This typically happens in the case of classes for which we gathered human-provided annotations, but that were discarded in the FSD50K release due to data scarcity (more specifically, they were merged with their parents). In other words, the main purpose of the `collection` format is to make available annotations for tiny classes. The format of these files in analogous to that of the files in `FSD50K.ground_truth/`. A couple of examples show the differences between **collection** and **ground truth** formats:
 
 `clip`:   `labels_in_collection`  -- `labels_in_ground_truth`

 `51690`: `Owl` -- `Bird,Wild_Animal,Animal`

 `190579`: `Toothbrush,Electric_toothbrush` -- `Domestic_sounds_and_home_sounds`

 In the first example, raters provided the label `Owl`. However, due to data scarcity, `Owl` labels were merged into their parent `Bird`. Then, labels `Wild_Animal,Animal` were added via label propagation (smearing). The second example shows one of the most extreme cases, where raters provided the labels `Electric_toothbrush,Toothbrush`, which both had few data. Hence, they were merged into Toothbrush's parent, which unfortunately is `Domestic_sounds_and_home_sounds` (a rather vague class containing a variety of children sound classes).

  **NOTE:** Labels in the collection format are not smeared.  
  **NOTE:** While in FSD50K's ground truth the vocabulary encompasses 200 classes (common for dev and eval), since the *collection* format is composed of raw annotations, the vocabulary here is much larger (over 350 classes), and it is slightly different in dev and eval.

For further questions, please contact eduardo.fonseca@upf.edu, or join the <a href="https://groups.google.com/g/freesound-annotator">freesound-annotator Google Group</a>.


## Download

The folders `FSD50K.ground_truth/`, `FSD50K.metadata/` and `FSD50K.doc/` are compressed into one zip file each. However, due to their large size, the folders `FSD50K.dev_audio/` and `FSD50K.eval_audio/` are split into several files. Specifically, `FSD50K.dev_audio/` is split into six files (note the last file is not `*.z06`, but `*.zip`): 

```
FSD50K.dev_audio.z01
FSD50K.dev_audio.z02
FSD50K.dev_audio.z03
FSD50K.dev_audio.z04
FSD50K.dev_audio.z05
FSD50K.dev_audio.zip
```

In this case, you first have to download all the files. Once downloaded, we merge all the files into one zip file called e.g. `unsplit.zip` in your local machine.

```
zip -s 0 FSD50K.dev_audio.zip --out unsplit.zip
```

Finally, this merged file is unzipped.

```
unzip unsplit.zip
```

Similar guidelines must be followed for the `FSD50K.eval_audio/` folder (only two zip files in this case).

## Baseline System

Several baseline systems for FSD50K are available at <a href="https://github.com/edufonseca/FSD50K_baseline">https://github.com/edufonseca/FSD50K_baseline</a>. The experiments are described in Sec 5 of our paper.


## References and links

[1] Jort F Gemmeke, Daniel PW Ellis, Dylan Freedman, Aren Jansen, Wade Lawrence, R Channing Moore, Manoj Plakal, and Marvin Ritter. "Audio set: An ontology and human-labeled dataset for audio events." In Proceedings of the International Conference on Acoustics, Speech and Signal Processing, 2017. [<a href="https://ai.google/research/pubs/pub45857">PDF</a>]

[2] Eduardo Fonseca, Jordi Pons, Xavier Favory, Frederic Font, Dmitry Bogdanov, Andres Ferraro, Sergio Oramas, Alastair Porter, and Xavier Serra. "Freesound Datasets: A Platform for the Creation of Open Audio Datasets." In Proceedings of the International Conference on Music Information Retrieval, 2017. [<a href="https://repositori.upf.edu/bitstream/handle/10230/33299/fonseca_ismir17_freesound.pdf">PDF</a>]


Companion site for FSD50K: <a href="https://annotator.freesound.org/fsd/release/FSD50K/">https://annotator.freesound.org/fsd/release/FSD50K/</a>  
Freesound Annotator: <a href="https://annotator.freesound.org/">https://annotator.freesound.org/</a>  
Freesound: <a href="https://freesound.org">https://freesound.org</a>  
Eduardo Fonseca's personal website: <a href="http://www.eduardofonseca.net/">http://www.eduardofonseca.net/</a>  
More datasets collected by us: <a href="http://www.eduardofonseca.net/datasets/">http://www.eduardofonseca.net/datasets/</a>


## Acknowledgments

The authors would like to thank everyone who contributed to FSD50K with annotations, and especially Mercedes Collado, Ceren Can, Rachit Gupta, Javier Arredondo, Gary Avendano and Sara Fernandez for their commitment and perseverance. The authors would also like to thank Daniel P.W. Ellis and Manoj Plakal from Google Research for valuable discussions. This work is partially supported by the European Union’s Horizon 2020 research and innovation programme under grant agreement No 688382 <a href="https://www.audiocommons.org/">AudioCommons</a>, and two Google Faculty Research Awards <a href="https://ai.googleblog.com/2018/03/google-faculty-research-awards-2017.html">2017</a> and <a href="https://ai.googleblog.com/2019/03/google-faculty-research-awards-2018.html">2018</a>, and the Maria de Maeztu Units of Excellence Programme (MDM-2015-0502).
