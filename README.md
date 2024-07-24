# CEBRA-Demo

![Python Version](https://img.shields.io/badge/python-3.9.7%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Documentation Coverage](https://img.shields.io/badge/documentation-100%25-brightgreen)

This is a repository documenting some of the analyses performed with CEBRA on neuromodulator data acquired using fiber-photometry. 
Specifically, the data is 4 neuromodulator's recorded simultaneously from the Nuclues Acumbens (there exists data from other regions, but these notebooks are using data from there).

This data can be found in the Code Ocean Capsule associated with this repository. If you run it in that capsule (this is the suggested way), no changes to the data loading are needed but you will need to start by installing the package within your environment (see step 1 below). If you clone the repository from git, you'll need to:
- install within a conda environment:
  - git clone https://github.com/BrianNGitahi/CEBRA_Pack
  - cd CEBRA_Pack
  - pip install .
- download the data from the data folder in the code ocean capsule: https://codeocean.allenneuraldynamics.org/capsule/2441328/tree?cw=true (to avoid path errors, I suggest you create a similar folder structure in your repository)

For a quick primer on CEBRA see the notebooks in the CEBRA_Pack repository, also linked on this webpage:https://brianngitahi.github.io/. 


NB: 
- If you'd like to understand how the input data has been formatted, see the notebook: Signal_rewarded_unrewarded.ipynb 
- If you have trouble installing CEBRA you can refer also to this page: https://cebra.ai/docs/installation.html

