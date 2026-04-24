MovieGraphs Dataset
------
Dataset release for our paper published at CVPR 2018.

[📜 MovieGraphs: Towards Understanding Human-Centric Situations from Videos](https://arxiv.org/abs/1712.06761)

[🧑‍💻 Project page](https://moviegraphs.cs.toronto.edu/)


### Release updates
- March 2019 (Original release with python 2.7; shared over emails)
- June 2025 (python 3.10 support)


### Files concerning movies
- dvds.txt: links to Amazon DVDs
- movies_list.txt: just a list of movies
- split.json: train/validation/test splits


### Nomenclature of video scenes
- xxx: scene id, starting from 1
- yyyy: start shot for scene xxx starting from 1
- zzzz: end shot for scene xxx
- [moviegraphs_startend_frames](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/makarand_tapaswi_iiit_ac_in/ER94dZuIwblItB7EFQliPtwBFjhFRFskf_Pf4BRiwWoMAg?e=HcCyc1): first and last frame of every movie scene clip (tarball, please expand)
- [shot boundaries (videvents) and scene GT boundaries](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/makarand_tapaswi_iiit_ac_in/EWeYiN1CoTRFg0QeXDs0dc0B-zNcCw_lsOwiMoWb2Rd0Tg?e=dCXfEN)


### [June 2025] Updated release
- py3loader_new
    - `all_movies.pkl`: main pickle file containing the parsed graph annotations.
    - `GraphClasses.py`: main ClipGraph and MovieGraph python classes. There are many helper functions here, please read this.
    - `startup.py`: a little test to make sure everything is setup correctly. This should run without errors.
    - `tutorial.ipynb`: a small notebook that tests functionality of some parts of `GraphClasses.py`. If you run all cells, it should create a pdf file with all graphs of one movie.
- nx_code: copies and updates essential files from `networkx=1.10` that were required for `GraphClasses.py` to work properly with `python 3`.


### Videos
⚠️🎞️ Full videos are not shared and should be procured using the DVDs listed in `dvds.txt`. If you intend to use the dataset for non-commercial and academic research purposes, a 1 fps version can be downloaded [here](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/makarand_tapaswi_iiit_ac_in/EXLfOOllPXBArTmY_K9c4XABJ0CmOOBy88IA9W34L7aa4A?e=mRDZcC).

Subtitles can be downloaded [here](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/makarand_tapaswi_iiit_ac_in/EY5UCcXRgAdJteMhG-vucwwB6vfsatQ-LobaUfEf1OK5Mw?e=b0iJf7).


### [Deprecated] Original release (shared over emails)
Worked with py2.7 and networkx1.10

- py3loader
    - 2017-11-02-51-7637_py3.pkl: main pickle file containing the parsed graph annotations
    - GraphClasses.py: main ClipGraph and MovieGraph python classes
    - startup.py: a little test file to make sure things are working as expected


### Acknowledgement
Thanks to [Lakshmipathi Balaji](https://www.linkedin.com/in/lakshmipathi-balaji-a46183218/) for helping with the new release!

