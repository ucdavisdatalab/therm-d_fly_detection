# Contributing

[top]: #contributing


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

([back to top][top])


## Training the Model

> [!IMPORTANT]
>
> The model has already been trained, so it is not necessary to train the model
> in order to use the fly detection tools described above. That said, if new
> annotated data becomes available, additional training may improve accuracy.

The model is a [You Only Look Once (YOLO) v8][yolo] object detection model. It
was trained on the UC Davis [Farm Cluster][farm]. The node used has 2 AMD EPYC
7713 64-core CPUs, 1TB RAM, and a NVIDIA A100 GPU. Training for 200 epochs took
approximately 2 hours.

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
