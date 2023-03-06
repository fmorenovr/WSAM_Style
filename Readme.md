# WSAM_Style
Visual explanations of Style augmentation influence.

## Installation

Install conda and then:

* Once installed conda, just run `conda env create --name wsam --file requirements.yaml `

or by pip and python env:

```
python3 -m venv wsam
source ./bin/activate
pip list --format=freeze > requirements.txt
python3 -m pip install -r requirements.txt 
```

## Data

* To download models, click [here](https://drive.google.com/file/d/1mbnimFjilKmmd9Wjt2RHlatrhVwdJ8q8/view?usp=share_link).
* Save all models into `models/` directory.
* Run `python download_STL-10.py` to download STL-10 dataset.

### Citation

```
@conference{visapp23,
author={Felipe Moreno{-}Vera. and Edgar Medina. and Jorge Poco.},
title={WSAM: Visual Explanations from Style Augmentation as Adversarial Attacker and Their Influence in Image Classification},
booktitle={Proceedings of the 18th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications - Volume 5: VISAPP,},
year={2023},
pages={830-837},
publisher={SciTePress},
organization={INSTICC},
doi={10.5220/0011795400003417},
isbn={978-989-758-634-7},
issn={2184-4321},
}
```
