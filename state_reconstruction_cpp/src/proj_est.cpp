#include "proj_est.hpp"

#include <fstream>
#include <pybind11/numpy.h>

// Projector application

void sliceLocalImages(const Eigen::Array<double,-1,-1,Eigen::RowMajor>& fullImage, std::vector<Image>& images)
{
    for(Image& image : images)
    {
        image.image = fullImage(Eigen::seq(image.X_min,image.X_max), Eigen::seq(image.Y_min,image.Y_max));
    }
}


std::vector<Image> getLocalImages(std::vector<Eigen::Vector2f> coords, const Eigen::Array<double,-1,-1,Eigen::RowMajor>& image, Eigen::Array2i shape, int psf_supersample)
{
    /*Extracts image subregions and subpixel shifts.

    Parameters
    ----------
    X, Y : `Array[1, float]`
        (Fractional) center positions in image coordinates.
    image : `Array[2, float]`
        Full-sized image.
    shape : `(int, int)`
        Shape of subregion images.
    psf_supersample : `int`
        Supersampling size
        (used to convert fractional positions to subpixel shifts).

    Returns
    -------
    ret : `dict(str->Any)`
        Returns a dictionary containing the following items:
    image : `np.ndarray(3, float)`
        Subregion images. Dimension: `[n_subregions, {shape}]`.
    X_int, Y_int : `np.ndarray(1, int)`
        Rounded PSF center coordinates.
    X_min, X_max, Y_min, Y_max : `np.ndarray(1, int)`
        Rounded PSF rectangle corners.
    dx, dy : `np.ndarray(1, int)`
        Subpixel shifts.*/

    std::vector<Image> localImages;
    for(Eigen::Vector2f& coord : coords)
    {
        Image imageN;
        imageN.X_int = std::round(coord[0]);
        imageN.Y_int = std::round(coord[1]);
        imageN.dx = std::round((coord[0] - imageN.X_int) * psf_supersample);
        imageN.dy = std::round((coord[1] - imageN.Y_int) * psf_supersample);
        imageN.X_min = imageN.X_int - shape[0] / 2;
        imageN.Y_min = imageN.Y_int - shape[1] / 2;
        imageN.X_max = imageN.X_min + shape[0] - 1;
        imageN.Y_max = imageN.Y_min + shape[1] - 1;
        localImages.push_back(imageN);
    }
    sliceLocalImages(image, localImages);
    return localImages;
}

std::vector<float> apply_projectors(std::vector<Image> localImages, py::object& projector_generator)
{
    /*Applies subpixel-shifted projectors to subregion images.

    Parameters
    ----------
    local_images : `np.ndarray(3, float)`
        Subregion images. Dimension: `[n_subregions, {shape}]`.
    projector_generator : `ProjectorGenerator`
        Projector generator object.

    Returns
    -------
    emissions : `np.ndarray(1, float)`
        Projected results. Dimensions: `[n_subregions]`.*/

    std::vector<float> emissions;
    bool projCacheBuilt = projector_generator.attr("proj_cache_built").cast<bool>();
    if(!projCacheBuilt)
    {
        projector_generator.attr("setup_cache")();
    }
    int psfSupersample = projector_generator.attr("psf_supersample").cast<int>();
    for(Image& localImage : localImages)
    {
        Eigen::VectorXd imagedata(Eigen::Map<Eigen::VectorXd>(localImage.image.data(), localImage.image.size()));
        py::array_t<double> projs = projector_generator.attr("proj_cache").cast<py::array_t<double>>();
        const ssize_t *shape = projs.shape();
        projs = projs.reshape(std::vector<int>({(int)(shape[0]), (int)(shape[1]), -1}));
        int xidx = localImage.dx % psfSupersample;
        int yidx = localImage.dy % psfSupersample;

        py::array imageProj = projs[py::make_tuple(xidx, yidx, py::ellipsis())];
        py::buffer_info info = imageProj.request();
        double *ptr = static_cast<double*>(info.ptr);
        Eigen::VectorXd projEigen = Eigen::Map<Eigen::VectorXd>(ptr, imageProj.size());

        emissions.push_back(imagedata.dot(projEigen));
    }
    return emissions;
}