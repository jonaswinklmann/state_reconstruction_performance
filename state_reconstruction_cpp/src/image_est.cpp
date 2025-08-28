#include "image_est.hpp"

#include <numeric>
#include <tuple>

/*Image preprocessing

Scales raw images to account for sensitivity inhomogeneities and
removes outliers.*/

void prescale_image(py::EigenDRef<Eigen::Array<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> fullImage, 
    const Eigen::Array<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>& scale)
{
    /*Scales an image by a sensitivity array.

    Parameters
    ----------
    im : `Array[2, float]`
        Raw image.
    scale : `Array[2, float]`
        Sensitivity scale. Must have same shape as `im`.

    Returns
    -------
    im_new : `Array[2, float]`
        Scaled image.*/

    fullImage /= scale;
}

std::tuple<Eigen::Array2d, Eigen::Array2i, double> analyze_image_outlier(
    py::EigenDRef<Eigen::Array<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> fullImage, 
    std::optional<Eigen::Array2i> outlier_size, double min_ref_val = 5)
{
    /*Analyzes an image for negative and positive outliers.

    Parameters
    ----------
    im : `Array[2, float]`
        Raw image.
    outlier_size : `(int, int)`
        Expected size of potential outliers.
    min_ref_val : `float`
        Minimum reference value to be considered valid outlier.

    Returns
    -------
    outlier_ratios : `np.ndarray([float, float])`
        Ratio between (background-subtracted) image minimum/maximum
        and reference, which is the `product(outlier_size)`-th
        smallest/largest image value.
        The background is the image median.
    outlier_idxs : `[(int, int), (int, int)]`
        Image coordinates of minimum/maximum pixel:
        `[(xmin, ymin), (xmax, ymax)]`.
    ar_bg : `float`
        Background (i.e. median) of image.*/

    if(!outlier_size.has_value())
    {
        outlier_size = Eigen::Array2i();
        outlier_size.value() << 5, 5;
    }

    std::vector<int> order(fullImage.size());
    std::iota(order.begin(), order.end(), 0);
    stable_sort(order.begin(), order.end(), [&fullImage](size_t i1, size_t i2) {return fullImage.data()[i1] < fullImage.data()[i2];});

    int prod = outlier_size.value()[0] * outlier_size.value()[1];
    double ar_min = fullImage.data()[order[0]];
    double ar_ref_min = fullImage.data()[order[prod]];
    double ar_max = fullImage.data()[order[fullImage.size() - 1]];
    double ar_ref_max = fullImage.data()[order[fullImage.size() - 1 - prod]];
    double ar_bg = fullImage.data()[order[fullImage.size() / 2]];
    Eigen::Array2d outlier_ratios;
    outlier_ratios << (ar_bg - ar_min) / std::max(min_ref_val, ar_bg - ar_ref_min),
        (ar_max - ar_bg) / std::max(min_ref_val, ar_ref_max - ar_bg);
    Eigen::Array2i outlier_idxs;
    outlier_idxs << order[0], order[fullImage.size() - 1];
    return std::make_tuple(outlier_ratios, outlier_idxs, ar_bg);
}

void remove_image_outlier(py::EigenDRef<Eigen::Array<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> fullImage, int outlier_idx, 
    double val, std::optional<Eigen::Array2i> outlier_size = std::nullopt)
{
    /*Removes an outlier from an image.

    Parameters
    ----------
    im : `Array[2, float]`
        Raw image.
    outlier_idx : `(int, int)`
        Image coordinates of central outlier pixel.
    outlier_size : `(int, int)`
        Removal area around outlier.
    val : `float`
        Replacement value.

    Returns
    -------
    im_new : `Array[2, float]`
        Copy of image with outlier area set to `val`.*/

    if(!outlier_size.has_value())
    {
        outlier_size = Eigen::Array2i();
        outlier_size.value() << 3, 3;
    }

    int xCoord = outlier_idx / (int)fullImage.cols();
    int yCoord = outlier_idx % (int)fullImage.cols();
    for(int x = std::max(0, xCoord - outlier_size.value()[0] / 2); 
        x < xCoord + (outlier_size.value()[0] + 1) / 2; x++)
    {
        for(int y = std::max(0, yCoord - outlier_size.value()[1] / 2); 
            y < yCoord + (outlier_size.value()[1] + 1) / 2; y++)
        {
            fullImage.data()[x * fullImage.cols() + y] = val;
        }
    }
    return;
}

std::tuple<int, Eigen::Array2d, Eigen::Array2i> process_image_outliers_recursive(
    py::EigenDRef<Eigen::Array<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> fullImage, 
    std::optional<double> max_outlier_ratio = std::nullopt, double min_ref_val = 5, 
    std::optional<Eigen::Array2i> outlier_size = std::nullopt, int max_its = 2, int it = 0)
{
    /*Recursively checks for outliers and enlarges the outlier size if necessary.

    Parameters
    ----------
    im : `Array[2, float]`
        Raw image.
    max_outlier_ratio : `float` or `None`
        Maximum allowed outlier ratio.
        If `None`, does not apply removal.
    min_ref_val : `float`
        Minimum reference value to be considered valid outlier.
    outlier_size : `(int, int)`
        Removal area around outlier.
    max_its : `int`
        Maximum number of iterations.
    _it : `int`
        Current iteration.

    Returns
    -------
    res : `dict(str->Any)`
        Analysis results containing the items:
    image : `Array[2, float]`
        Image with outliers removed.
    iterations : `int`
        Number of iterations applied.
        (`iterations == 1` means no recursive call.)
    outlier_ratios : `np.ndarray([float, float])`
        Minimum/maximum outlier ratios for last iteration.
    outlier_size : `(int, int)`
        Removal area for last iteration.*/

    if(!outlier_size.has_value())
    {
        outlier_size = Eigen::Array2i();
        outlier_size.value() << 5, 5;
    }

    // Analyze outlier
    auto [outlier_ratios, outlier_idxs, bg] = analyze_image_outlier(fullImage, outlier_size, min_ref_val);
    
    if (!max_outlier_ratio.has_value() || (outlier_ratios[0] < max_outlier_ratio.value() && 
        outlier_ratios[1] < max_outlier_ratio) || it >= max_its)
    {
        return std::make_tuple(it, outlier_ratios, outlier_size.value());
    }
    else
    {
        // Remove outliers
        for(int i = 0; i < 2; i++)
        {
            if (outlier_ratios[i] > max_outlier_ratio)
            {
                remove_image_outlier(fullImage, outlier_idxs[i], bg, outlier_size.value());
            }
        }
        // Call function to get analysis
        return process_image_outliers_recursive(fullImage, max_outlier_ratio.value(), 
            min_ref_val, outlier_size.value(), max_its, it+1);
    }
}

    /*def get_attr_str():
        s = []
        if self.scale is not None:
            _scale_mean = f"{np.mean(self.scale):.2f}"
            _scale_shape = ("scalar" if np.isscalar(self.scale)
                            else str(self.scale.shape))
            s.append(f" → scale: mean: {_scale_mean}, shape: {_scale_shape}")
        for k in ["outlier_size", "max_outlier_ratio", "outlier_iterations"]:
            s.append(f" → {k}: {str(getattr(self, k))}")
        return "\n".join(s)

    def __str__(self):
        return f"{self.__class__.__name__}:\n{self.get_attr_str()}"

    def __repr__(self):
        s = f"<'{self.__class__.__name__}' at {hex(id(self))}>"
        return f"{s}\n{self.get_attr_str()}"*/

Eigen::Array2d ImagePreprocessor::process_image(py::EigenDRef<Eigen::Array<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> image) const
{
    /*Performs image preprocessing (scaling and outlier detection).

    Parameters
    ----------
    im : `Array[2, float]`
        Raw image.
    ret_success : `bool`
        Whether to return a success flag.

    Returns
    -------
    im : `Array[2, float]`
        Processed image.
    outlier_ratio : `bool`
        Outlier ratio.*/

    // Prescale image
    if(this->scale.has_value() && this->scale->size() == image.size())
    {
        prescale_image(image, *this->scale);
    }
    // Process outliers
    auto [iterations, outlier_ratios, outlier_size] = process_image_outliers_recursive(image, 
        this->max_outlier_ratio, this->outlier_min_ref_val, this->outlier_size,this->outlier_iterations);
        
    /*s_outl_ratios = misc.cv_iter_to_str(outlier_ratios, fmt="{:.1f}")
    s_outl_size = misc.cv_iter_to_str(outlier_size, fmt="{:.0f}")
    _msg = (
        f"(ratios: {s_outl_ratios:s}, size: {s_outl_size}, "
        f"iterations: {outlier_iterations:d})"
    )
    if outlier_iterations == 1:
        self.LOGGER.debug(f"process_image: No outlier detected {_msg}")
    else:
        if np.any(outlier_ratios > self.max_outlier_ratio):
            self.LOGGER.warning(
                f"process_image: Outlier removal failed {_msg}"
            )
        else:
            self.LOGGER.info(f"process_image: Outlier detected {_msg}")*/
    return outlier_ratios;
}