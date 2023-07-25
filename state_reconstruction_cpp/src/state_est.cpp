#include "state_est.hpp"

#include <pybind11/numpy.h>
#include <optional>
#include <fstream>

#include "linear.hpp"
#include "proj_est.hpp"
#include "trafo_est.hpp"

std::vector<double> StateEstimator::constructLocalImagesAndApplyProjectors(
        const py::EigenDRef<const Eigen::Array<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>>& image,
        Eigen::Array2i sitesShape, py::object& trafoPy, Eigen::Array2i projShape, int psfSupersample, py::object& projector_generator,
        py::EigenDRef<Eigen::Array<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>>& emissions)
{
    std::vector<Eigen::Vector2i> coords;
    for(int i = 0; i < sitesShape[0]; i++)
    {
        for(int j = 0; j < sitesShape[1]; j++)
        {
            Eigen::Vector2i vec;
            vec << i,j;
            coords.push_back(vec);
        }
    }
    std::vector<Eigen::Array2i> imageRect;
    Eigen::Array2i rect;
    rect << projShape[0], image.rows() - projShape[0];
    imageRect.push_back(rect);
    rect = Eigen::Array2i();
    rect << projShape[1], image.cols() - projShape[1];
    imageRect.push_back(rect);

    AffineTrafo2D trafo(trafoPy);
    auto [emissionCoords, originCoords] = trafo.filter_origin_coords_within_target_rect(coords, imageRect);
    auto localImages = getLocalImages(emissionCoords, image, projShape, psfSupersample);

    // Apply projectors and embed local images
#ifdef CUDA
    auto localEmissions = apply_projectors_gpu(image, localImages, projector_generator);
#else
    auto localEmissions = apply_projectors(localImages, projector_generator);
#endif
    for(size_t i = 0; i < localEmissions.size(); i++)
    {
        emissions(originCoords[i][0], originCoords[i][1]) = localEmissions[i];
    }
    return localEmissions;
}