# Installation

Requires Python 3.9 or later.

```bash
pip install dhybridrpy
```

This installs the Python package, its dependencies, and the `dplot` command.
For MP4 output, also install [FFmpeg](https://ffmpeg.org/) and make `ffmpeg`
available on your `PATH`.

## Install from source

A source checkout includes sample simulation output and an example notebook:

```bash
git clone https://github.com/AstroplasmasUChicago/dhybridrpy.git
cd dhybridrpy
pip install .
```

To use the sample data, open `examples/examples.ipynb` and run its cells
with `examples/` as the working directory.
