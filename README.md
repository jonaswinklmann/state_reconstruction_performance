# State reconstruction

Quantum state reconstruction from microscopic fluorescence images of atoms on a lattice.

Includes the following functionalities:
* Simulate noisy fluorescence images
* Extract subpixel point spread functions
* Extract lattice vectors
* Reconstruct single-site occupation

The full documentation can be found [here](https://david-wei.github.io/state_reconstruction).

## Installation

Clone this repository:

```
git clone https://www.github.com/david-wei/state_reconstruction.git
```


Install the library:

```
pip install ./state_reconstruction
```


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
