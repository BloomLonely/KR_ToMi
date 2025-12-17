# KR_ToMi: Korean Theory of Mind Dataset

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

Korean adaptation of the ToMi (Theory of Mind) dataset for evaluating false belief understanding in Korean language models.

## Project Overview

This project adapts the [ToMi dataset](https://github.com/facebookresearch/ToMi) from Facebook Research to Korean, enabling comparative analysis of Theory of Mind capabilities between English and Korean language models (e.g., EXAONE).

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Generate English Dataset

```bash
python scripts/generate_dataset.py --num-stories 1000 --out-dir data/english
```

### Generate Korean Dataset (Coming Soon)

```bash
python scripts/generate_dataset.py --num-stories 1000 --out-dir data/korean --lang ko
```

## Project Structure

```
KR_ToMi/
├── data/                      # Generated datasets
│   ├── english/              # English ToMi dataset
│   │   ├── fb_all_train.txt
│   │   ├── fb_all_train.trace
│   │   ├── fb_all_val.txt
│   │   ├── fb_all_val.trace
│   │   ├── fb_all_test.txt
│   │   └── fb_all_test.trace
│   └── korean/               # Korean ToMi dataset (to be generated)
├── src/                       # Source code
│   ├── __init__.py
│   ├── world.py              # Entity management
│   ├── oracle.py             # Belief tracking system
│   ├── story.py              # Story generation logic
│   ├── actions.py            # Action types and templates
│   └── world.json            # Entity definitions
├── scripts/                   # Executable scripts
│   └── generate_dataset.py   # Dataset generation script
├── docs/                      # Documentation
│   ├── README.md             # Original ToMi README
│   └── CLAUDE.md             # Project analysis and guide
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Data

The data follows the same format and uses the same models as the [`tom-qa-dataset`](https://github.com/kayburns/tom-qa-dataset) repository.  We do include one supplementary file for each `*.txt` file that classifies the story/question type in each example (which contains a `.trace` extension).  Each line in a trace file contains a high level abstraction of the story as well as a classification of the question and a classification of the story.  Story types can be one of:

1. `true_belief` - All agents observed all actions
2. `false_belief` - An agent failed to observe an action
3. `second_order_false_belief` - An agent has a false belief about another agent's set of beliefs
 
Question types can be one of:

1. `first_order_(0|1)_tom` - A first order false belief question in a story where a false belief situaion has been established
2. `first_order_(0|1)_no_tom` - A first order false belief question in a story where the agent in question observed all actions
3. `second_order_(0|1)_tom` - A second order false belief question in a story where a second order false belief situation has been established
4. `second_order_(0|1)_no_tom` - A second order false belief question in a story where the agent in question does not have a second order false belief
8. `reality` - A control question (ex: "Where is object *x* now?")
9. `memory` - A control question (ex: "Where was object *x* at the beginning?")


## References

If you find this code useful for your research, please cite the following paper in your publication:


```bibtex
@inproceedings{le-etal-2019-revisiting,
    title = "Revisiting the Evaluation of Theory of Mind through Question Answering",
    author = "Le, Matthew  and
      Boureau, Y-Lan  and
      Nickel, Maximilian",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-1598",
    doi = "10.18653/v1/D19-1598",
    pages = "5872--5877"
}
```

## License 

This code is licensed under [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

![](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)