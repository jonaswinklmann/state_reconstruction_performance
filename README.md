# State reconstruction
Modified version of [David Wei's solution](https://github.com/david-wei/state_reconstruction)
This README is also adopted and only changed where necessary.

Fluorescence emission state reconstruction from microscopic fluorescence images of atoms on a lattice.

Includes the following functionalities:
* Simulate noisy fluorescence images
* Extract subpixel point spread functions
* Extract lattice vectors
* Reconstruct single-site occupation

The full documentation can be found [here](https://david-wei.github.io/state_reconstruction).

## Installation

### Clone this repository:

```
git clone https://github.com/jonaswinklmann/state_reconstruction_performance.git
```

### Build the underlying C library depending on the target device:

Linux:
```
cd state_reconstruction_performance/state_reconstruction_cpp/
make all cpu
cd ../../
```
Windows:
1. Open Visual Studio
2. Open project property pages. Make sure configuration is set to "All Configurations". Adjust "C/C++ -> General -> Additional Include Directories", "Linker -> General -> Output File", "Linker -> General -> Additional Library Directories", "Linker -> Input -> Additional Dependencies", and "Build Events -> Post-Build Event -> Command Line" according to your Python installation".
3. Set the Solution Configuration Selector to "Release" (CPU) or "GPU" depending on the target machine (GPU requires a CUDA capable device".
4. Build the project by right-clicking on the project and selecting "Build" or "Rebuild".
5. Make sure the top-level "state_reconstruction" directory contains the resulting ".pyd" file.

### Install the library:

```
pip install ./state_reconstruction
```

For Linux, there is also a script called compileCppAndInstallPip.sh that combines building of the C library and pip installation.

## Dependencies

* [libics](https://www.github.com/david-wei/libics)


## Getting started

The package API is defined in the file [api.py](./state_reconstruction/api.py).

Tutorials are provided as Jupyter notebooks in the directory [scripts](./scripts):
* The notebook [`image_generation`](./scripts/image_generation.ipynb) demonstrates how to
  * set up coordinate transformations between site and image coordinates,
  * define point spread function generator objects,
  * generate noisy fluorescence images.
* The notebook [`psf+trafo_estimation`](./scripts/psf+trafo_estimation.ipynb) uses the files generated in `image_generation` to demonstrate how to
  * estimate a pixel-supersampled point spread function,
  * fit an affine transformation between site and image coordinates.
* The notebook [`state_reconstruction`](./scripts/state_reconstruction.ipynb) uses the files generated in `image_generation` to demonstrate how to set up the reconstruction process, including
  * image preprocessing for outlier removal,
  * image-to-site-space projector generation,
  * histogram analysis,
  * usage of the main `StateEstimator` object.
