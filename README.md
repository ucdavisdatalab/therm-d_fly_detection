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

Contents:

* [Usage](#usage)
    - [Data Format](#data-format)
    - [Workflow](#workflow)
    - [Output Format](#output-format)
* [Directories and Files](#directories-and-files)
* [Installation](#installation)
* [Contributing](#contributing)
* [Training the Model](#training-the-model)

[top]: #hamada-fly-behavior-startup-project


## Usage

### Data Format

The fly detection tools assume a standard format for data sets:

```
YYYY-MM-DD_apparatus/       the data set directory
├── photos/                 original photos, in JPEG format
└── temperatures.xlsx       temperature spreadsheet, in XLSX format
```

The data set directory must be named with the date and apparatus name. The
photos in `photos/` can have any name but must have extension `.jpg` or
`.jpeg`. The `.xlsx` file can have any name. In general, it's not a good idea
to put spaces in file names (use at your own risk!).

For example, one of the data sets we used during development had this format:

```
2023-07-19_biden
├── photos/
│   ├── 2023-07-19_biden_photos_01.JPG
│   ├── 2023-07-19_biden_photos_02.JPG
│   ├── 2023-07-19_biden_photos_03.JPG
│   ├── 2023-07-19_biden_photos_04.JPG
│   ├── ...
│   └── 2023-07-19_biden_photos_42.JPG
└── 230719_Biden_Leia_template.xlsx
```

([back to top][top])

### Workflow

For an appropriately formatted data set, the fly detection workflow consists of
two steps:

1.  **Arena Registration** (the `register` command). For each image in a data
    set's `photos/` subdirectory:
    1)  Standardize the brightness and contrast.
    2)  Find the registration marks.
    3)  Rotate as needed so that the image is not upside-down or sideways.
    4)  Correct for perspective distortion so that the registration marks form
        a perfect rectangle.
    5)  Crop to the rectangle.
    6)  Save the cropped image in the data set's `arenas/` subdirectory.

2.  **Fly Detection** (the `detect` command). For each image in a data set's
    `arenas/` subdirectory:
    1)  Use the model to predict fly locations as bounding boxes.
    2)  Remove bounding boxes with too much overlap (the model typically makes
        some redundant predictions).
    3)  Estimate the temperature at each fly's location.
    4)  Save the predicted fly locations and estimated temperatures to a `.csv`
        or `.parquet` file in the data set directory.

These steps are implemented as two different command-line commands (`register`
and `detect`, respectively).

All of the command-line commands have only one argument: a path to a [TOML][]
configuration file. TOML is a plain-text configuration file format designed to
be easy to read and write. An example configuration file is provided in this
repo at [`configs/defaults.toml`][defaults]. A copy of the file with long-form
documentation is provided in this repo at
[`configs/defaults-long-comments.toml`][defaults-long]. You can open and edit
TOML files with a text editor such as [Notepad++][], [TextEdit][], or [nano][].

[defaults]: configs/defaults.toml
[defaults-long]: configs/defaults-long-comments.toml
[TOML]: https://toml.io/
[Notepad++]: https://notepad-plus-plus.org/
[TextEdit]: https://support.apple.com/guide/textedit/welcome/mac
[nano]: https://en.wikipedia.org/wiki/GNU_nano

The TOML config file contains settings for each command, as well as physical
measurements (in centimeters) for each apparatus. We recommend that you create
a new TOML config file for each data set, so that you have a record of the
settings you used to process each data set. The easiest way to do this is to
copy `configs/defaults.toml` or another config file and then edit as needed.

In the TOML config file, the most important setting is `data_path`, which
should be set to the path to the data set directory. This setting and others
are documented in `configs/defaults.toml`.

Once you've created a TOML config file, for example
`configs/2023-07-19_biden.toml`, you can run arena registration. In a terminal,
navigate to the repo (with `cd`) and make sure the `fly` environment is
activated (`mamba activate fly`). Then run:

```sh
python -m src register configs/2023-07-19_biden.toml
```

This will create an `arenas/` subdirectory in the data set directory. You can
inspect the images in `arenas/` to check that the arenas were detected
correctly.

Arena detection typically fails if the registration marks are covered in the
original photo or the original photo has poor brightness or contrast. As a
failsafe, you can manually specify the pixel coordinates of the apparatus
corners in the TOML configuration file with the `arena` setting. An example of
this is provided in [`configs/2023-08-07_biden.toml`][manual-arena].

[manual-arena]: configs/2023-08-07_biden.toml

Next, you can run fly detection. In the terminal, run:

```sh
python -m src detect configs/2023-07-19_biden.toml
```

This will create a `predictions.csv` file and `predictions/` subdirectory in
the data set directory. The `predictions/` subdirectory contains visualizations
of the predicted flies as JPEG images (one for each arena image). The images
show each detected fly's bounding box, an identification number (ID) for the
box, and grid lines every 0.5 degrees Celsius.

Note that box IDs is not linked across images, so box `1` for the first image
in a data set does not necessarily enclose the same fly as box `1` for the
second image. Box `id` is only provided as a way to easily remove incorrect
boxes.

The format of `predictions.csv` is described in the next section.

([back to top][top])

### Output Format

The `predictions.csv` file created by the `detect` command has one row for each
detected fly. The model detects a bounding box around each fly, so the columns
contain data about the bounding box. The columns are:

Column           | Description
---------------- | -----------
`id`             | identification number for the box (within the image)
`x_px`           | x-coordinate of the box center, in pixels
`y_px`           | y-coordinate of the box center, in pixels
`width_px`       | width of the box, in pixels
`height_px`      | height of the box, in pixels
`confidence`     | confidence score for the box (from 0 lowest confidence to 1 highest confidence)
`path`           | file path to arena image
`arena_width_px` | width of the arena, in pixels
`arena_height_px`| height of the arena, in pixels
`x_cm`           | x-coordinate of the box center, in centimeters
`y_cm`           | y-coordinate of the box center, in centimeters
`temperature`    | estimated temperature at the box center, in degrees Celsius

The file is a comma-separated values (CSV) file, which can be read and analyzed
with data analysis software such as Excel, Tableau, Python, and R. The TOML
config file also provides a setting to save the file in [Parquet][] format.
Parquet is an open-standard for data exchange that provides [several
benefits][benefits-parquet] over CSV files.

[Parquet]: https://parquet.apache.org/
[benefits-parquet]: https://ucdavisdatalab.github.io/workshop_reproducible_research/chapters/03_case_by_case_core.html#use-file-formats-effectively

([back to top][top])


## Directories and Files

The directories and files in this repository are:

```
configs/      TOML configurations for commands
data/         Data sets (files > 1MB go on Google Drive)
models/       Deep learning models for fly detection
notebooks/    Jupyter notebook source files (exploratory code)
src/          Python source code

.gitignore    Settings file for git
README.md     This file
fly.yml       Main Conda environment (with OpenCV, etc)
fly-dev.yml   Conda environment for development
tess.yml      Conda environment for Tesseract
```

Each `.md` file in `notebooks/` and `.py` file in `src/` has a brief
description at the top of the file. The `data/` and `models/` directories are
not included with the repo, but typically need to be created to use the tools.

([back to top][top])


## Installation

The fly detection tools are Unix command line tools, so some familiarity with
the command line will make installing and using them easier. You can learn more
about the Unix command line from DataLab's ["Introduction to the Unix Command
Line][intro-cmd] workshop reader.

[intro-cmd]: https://ucdavisdatalab.github.io/workshop_introduction_to_the_command_line/

Make sure your computer has git installed. You can learn more about git from
DataLab's ["Introduction to Version Control"][intro-vcs] workshop reader.

[intro-vcs]: https://ucdavisdatalab.github.io/workshop_introduction_to_version_control/

Use `git clone` to copy this repository from GitHub to your computer:

```sh
git clone git@github.com:datalab-dev/2023_project_hamada_fly.git
```

Change directories to the cloned repo:
```sh
cd 2023_project_hamada_fly/
```

Next, create a `models/` subdirectory:
```sh
mkdir models
```

Go to the [Google Drive][google] and download the file
`models/2023-08-25_fly-detection.onnx` to the `models/` subdirectory you just
created.

Make sure your computer has conda or mamba installed. You can learn more about
these tools from [this section][conda-reader] of DataLab's "Making Python
Projects & Environments Reproducible" workshop reader. We recommend installing
[miniforge][] (formerly known as "mambaforge") and using [mamba][] because it's
generally faster, and we provide mamba commands below. If you're using conda,
replace "mamba" with "conda" in the commands.

[miniforge]: https://github.com/conda-forge/miniforge
[mamba]: https://mamba.readthedocs.io/
[conda-reader]: https://ucdavisdatalab.github.io/workshop_intermediate_python/chapters/02_reproducible.html#what-s-an-environment

Use conda or mamba to recreate the Python environment required by the fly
detection tools:

```sh
mamba env create --file fly.yml
```

This will create an environment named `fly`. Finally, activate the environment:

```sh
mamba activate fly
```

Now you're ready to use the fly detection tools!

([back to top][top])


## Contributing

_This section is about how to contribute notebooks to the repo._

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

([back to top][top])


## Training the Model

The model has already been trained, so it is not necessary to train the model
in order to use the fly detection tools described above. That said, if new
annotated data becomes available, additional training may improve accuracy.

The model is a [You Only Look Once (YOLO) v8][yolo] object detection model. It
was trained on the UC Davis [Farm Cluster][farm]. The node has 64 CPUs, 256GB
RAM, and a NVIDIA A100 GPU. Training for 200 epochs took approximately 2 hours. 

[yolo]: https://github.com/ultralytics/ultralytics/
[farm]: https://www.hpc.ucdavis.edu/clusters

We created training data by [template matching][] 15-degree rotations of a
manually cropped fly to each image in these data sets:

* `2023-07-12_biden`
* `2023-07-19_biden`
* `2023-07-19_shiny`
* `2023-07-22_shiny`
* `2023-07-23_skywalker`
* `2023-07-26_shiny`
* `2023-08-09_biden`

We then manually corrected these annotations (adding boxes around undetected
flies and removing incorrect boxes) with the free [MakeSense][] annotation
software. This process is time-consuming, as template matching is often
inaccurate. Going forward, it is likely more efficient to manually-correct
predictions from the current fly detection model.

[template matching]: https://en.wikipedia.org/wiki/Template_matching
[MakeSense]: https://www.makesense.ai/

Before running template matching, create an `outputs/` subdirectory in the
repo, then go to the [Google Drive][google] and download the file
`outputs/fly_template.npz` to the subdirectory you just created.

The command to run template matching is:

```sh
python -m src detect configs/defaults.toml
```

As with other commands, the TOML config file contains settings such as which
data set to use. The resulting annotations are saved to the `match_labels/`
subdirectory of the data set directory.

After manually correcting the annotations in [MakeSense][], export the
annotations in YOLO format and unzip the resulting `.zip` file into a `labels/`
subdirectory of the data set directory.

You can prepare a training data set, combining annotations for several data
sets, with the command:

```sh
python -m src assemble configs/train.toml
```

See the TOML config file for settings.

<!--
TODO: Add instructions for setting up the training environment.
-->

Finally, to train the model, run:

```sh
python -m src.train configs/train.toml
```

When training is complete, the model will be saved in the `models/` directory
of the repo. Note that the training script was designed and tested for training
the original pretrained `YOLOv8x` model; it was not tested for additional
training of the fly detection model, so some editing may be necessary.

([back to top][top])
