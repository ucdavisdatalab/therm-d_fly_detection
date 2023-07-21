# Hamada Fly Behavior Startup Project

This repository contains code for Fly Behavior startup project in collaboration
with Fumika Hamada. The project is about studying the temperature preferences
of fruit flies (Drosophila) over a 24-hour period, in order to better
understand which genes mediate circadian rhythm. The goal of our collaboration
is automate the process of counting fruit flies and estimating their
temperatures in photos from experiments.

Links:

* [Google Drive][google]

[google]: https://drive.google.com/drive/folders/1FIguz398nbSabeofCjJUHQ59yog626J6

## File and Directory Structure

The directory structure for the project is:

```
env.yml       Main Conda environment (with OpenCV, etc)
tess.yml      Conda environment for Tesseract
README.md     This file
data/         Data sets (files > 1MB go on Google Drive)
docs/         Supporting documents
notebooks/    Jupyter and RMarkdown notebook source files
reports/      HTML or PDF reports generated from notebooks
└── figures/  Graphics and figures to be used in reporting
src/          Python source code
```

<!--
The files in the `data/` directory are:

```

```
-->

## Notebooks

Jupyter notebooks are stored in the repo in Markdown format (`.md`) via
[Jupytext][]. This makes it easier to see changes to the notebooks in version
control and also avoids committing large images to the repo.

The conda environments in the repo include Jupytext. Make sure one is installed
and active before running the commands below.

[Jupytext]: https://jupytext.readthedocs.io/en/latest/

Whenever you create a new Jupyter notebook, say `notebook.ipynb`, run this
command to make it a **paired notebook** and generate a corresponding
`notebook.md` file:

```sh
jupytext --set-formats 'ipynb,md' notebook.ipynb
```

The `notebook.md` file is the one to commit to the repo. Once a notebook is
paired, the two files will automatically be kept in sync as long as you run
Jupyter in an environment that has Jupytext installed.

If you over need to manually sync a paired notebook, the command is:

```sh
jupytext --sync notebook.ipynb
```
Be careful when doing this, because if you've somehow ended up with changes to
both `notebook.ipynb` and `notebook.md`, the older changes will be overwritten!


### Additional Jupytext Commands

You generally won't need to run these if you've followed the instructions
above.

To convert all `.ipynb` files to `.md`, run this command in the `notebooks/`
directory:

```sh
jupytext --to md *.ipynb
```

You can replace `*.ipynb` with a specific file name if you only want to convert
one file.

To convert `.md` to `.ipynb`, run this command:

```sh
jupytext --to ipynb *.md
```

See the [Jupytext CLI docs][jupytext-cli] for more info. There is also a
Jupytext extension for JupyterLab that can handle this process automatically.

[jupytext-cli]: https://jupytext.readthedocs.io/en/latest/using-cli.html

