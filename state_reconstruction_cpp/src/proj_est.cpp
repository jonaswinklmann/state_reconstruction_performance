#include "proj_est.hpp"

#include <pybind11/numpy.h>

#include "calcEmissions.hpp"

// Projector application

std::vector<Image> getLocalImages(std::vector<Eigen::Vector2d> coords, 
    const Eigen::Array<double,-1,-1,Eigen::RowMajor>& fullImage, Eigen::Array2i shape, int psf_supersample)
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
    for(Eigen::Vector2d& coord : coords)
    {
        int x_int = std::round(coord[0]);
        int y_int = std::round(coord[1]);
        int x_min = x_int - shape[0] / 2;
        int y_min = y_int - shape[1] / 2;
        int x_max = x_min + shape[0] - 1;
        int y_max = y_min + shape[1] - 1;
        Image imageN
        {
            .image = fullImage.data(),
            .offset = (size_t)(x_min * fullImage.cols() + y_min),
            .outerStride = (size_t)(fullImage.cols()),
            .innerStride = 1,
            .X_int = x_int,
            .Y_int = y_int,
            .X_min = x_min,
            .X_max = x_max,
            .Y_min = y_min,
            .Y_max = y_max,
            .dx = (int)(std::round((coord[0] - x_int) * psf_supersample)),
            .dy = (int)(std::round((coord[1] - y_int) * psf_supersample))
        };
        localImages.push_back(imageN);
    }
    return localImages;
}

std::vector<double> apply_projectors(std::vector<Image>& localImages, py::object& projector_generator)
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

    std::vector<double> emissions;
    bool projCacheBuilt = projector_generator.attr("proj_cache_built").cast<bool>();
    if(!projCacheBuilt)
    {
        projector_generator.attr("setup_cache")();
    }
    int psfSupersample = projector_generator.attr("psf_supersample").cast<int>();

    py::array_t<double> projs = projector_generator.attr("proj_cache").cast<py::array_t<double>>();
    const ssize_t *shape = projs.shape();
    projs = projs.reshape(std::vector<int>({(int)(shape[0]), (int)(shape[1]), -1}));

    for(Image& localImage : localImages)
    {
        int xidx = localImage.dx % psfSupersample;
        int yidx = localImage.dy % psfSupersample;

        py::array imageProj = projs[py::make_tuple(xidx, yidx, py::ellipsis())];
        py::buffer_info info = imageProj.request();
        double *ptr = static_cast<double*>(info.ptr);

        double sum = 0;
        int cols = localImage.X_max - localImage.X_min + 1;
        int pixelCount = cols * (localImage.Y_max - localImage.Y_min + 1);
        for(int i = 0; i < pixelCount; i++)
        {
            sum += localImage.image[localImage.offset + (i / cols) * 
                localImage.outerStride + i % cols] * ptr[i];
        }
        
        emissions.push_back(sum);
    }
    return emissions;
}

#ifdef CUDA
std::vector<double> apply_projectors_gpu(const Eigen::Array<double,-1,-1,Eigen::RowMajor>& fullImage, 
    std::vector<Image>& localImages, py::object& projector_generator)
{
    std::vector<double> emissions;
    bool projCacheBuilt = projector_generator.attr("proj_cache_built").cast<bool>();
    if(!projCacheBuilt)
    {
        projector_generator.attr("setup_cache")();
    }
    int psfSupersample = projector_generator.attr("psf_supersample").cast<int>();

    py::array_t<double> projs = projector_generator.attr("proj_cache").cast<py::array_t<double>>();
    const ssize_t *shape = projs.shape();
    projs = projs.reshape(std::vector<int>({(int)(shape[0]), (int)(shape[1]), -1}));
    shape = projs.shape();
    
    return calcEmissionsGPU(fullImage.data(), fullImage.rows(), fullImage.cols(), projs.data(), projs.size(), shape, localImages, psfSupersample);
}
#endif