#include "state_est.hpp"

#include <pybind11/numpy.h>
#include <optional>
#include <fstream>

#include "calcEmissions.hpp"
#include "linear.hpp"
#include "proj_est.hpp"
#include "trafo_est.hpp"
#include "trafo_gen.hpp"

AffineTrafo2D get_phase_shifted_trafo(
    const py::EigenDRef<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& image, py::object &projector_generator,
    Eigen::Array2d phase_ref_image, Eigen::Array2d phase_ref_site, py::object &trafoSiteToImage)
{
    /*Gets a phase-shifted lattice transformation from an image.

    Parameters
    ----------
    phase : `(float, float)`
        Directly specifies phase. Takes precedence over `image`.
    image : `Array[2, float]`
        Image from which to extract lattice transformation.
    method : `str`
        Method for determining lattice transformation:
        `"projection_optimization", "isolated_atoms"`.
    preprocess_image : `bool`
        Whether to preprocess image.

    Returns
    -------
    new_trafo : `AffineTrafo2d`
        Phase-shifted lattice transformation.*/

    // Image preprocessing
    // From emission variance maximization
    
    TrafoEstimator trafoEst;
    Eigen::VectorXd phase = trafoEst.get_trafo_phase_from_projections(image, projector_generator, phase_ref_image, phase_ref_site);

    // Construct phase-shifted trafo
    return get_trafo_site_to_image(AffineTrafo2D(trafoSiteToImage), 
        std::nullopt, std::nullopt, phase_ref_image, phase_ref_site, phase);
}

StateEstimator::StateEstimator() : imgPreproc(std::nullopt) {}

std::vector<double> StateEstimator::constructLocalImagesAndApplyProjectorsPyTrafo(
    const py::EigenDRef<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &image,
    Eigen::Array2i sitesShape, py::object &trafoPy, Eigen::Array2i projShape, int psfSupersample, py::object &projector_generator,
    py::EigenDRef<Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> emissions)
{
    AffineTrafo2D trafo(trafoPy);
    return this->constructLocalImagesAndApplyProjectors(image, sitesShape, trafo, projShape, psfSupersample, projector_generator, emissions);
}

std::vector<double> StateEstimator::constructLocalImagesAndApplyProjectors(
    const py::EigenDRef<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &image,
    Eigen::Array2i sitesShape, AffineTrafo2D &trafo, Eigen::Array2i projShape, int psfSupersample, py::object &projector_generator,
    py::EigenDRef<Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> emissions)
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
    rect << projShape[0], (int)image.rows() - projShape[0];
    imageRect.push_back(rect);
    rect = Eigen::Array2i();
    rect << projShape[1], (int)image.cols() - projShape[1];
    imageRect.push_back(rect);

    auto [emissionCoords, originCoords] = trafo.filter_origin_coords_within_target_rect(coords, imageRect);
    auto localImages = getLocalImages(emissionCoords, image, projShape, psfSupersample);

    // Apply projectors and embed local images
#ifdef CUDA
    EmissionCalculatorCUDA& emissionCalcGPU = EmissionCalculatorCUDA::getInstance();
    emissionCalcGPU.allocatePerImageBuffers(localImages.size());
    auto localEmissions = emissionCalcGPU.calcEmissionsGPU(localImages, psfSupersample);
#else
    auto localEmissions = apply_projectors(localImages, projector_generator);
#endif
    for(size_t i = 0; i < localEmissions.size(); i++)
    {
        emissions(originCoords[i][0], originCoords[i][1]) = localEmissions[i];
    }
    return localEmissions;
}

void StateEstimator::init()
{
#ifdef CUDA
    EmissionCalculatorCUDA::getInstance().initGPUEnvironment();
#endif
}

void StateEstimator::loadProj(py::object& prjgen)
{
#ifdef CUDA
    EmissionCalculatorCUDA::getInstance().loadProj(prjgen);
#endif
}

void StateEstimator::setImagePreProcScale(const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &scale, 
    std::optional<Eigen::Array2i> outlier_size, double max_outlier_ratio, double outlier_min_ref_val, int outlier_iterations)
{
    this->imgPreproc.emplace(ImagePreprocessor(scale, outlier_size, max_outlier_ratio, outlier_min_ref_val, outlier_iterations));
}

void StateEstimator::setImagePreProc(std::optional<Eigen::Array2i> outlier_size, double max_outlier_ratio, 
    double outlier_min_ref_val, int outlier_iterations)
{
    this->imgPreproc.emplace(ImagePreprocessor(outlier_size, max_outlier_ratio, outlier_min_ref_val, outlier_iterations));
}

std::tuple<std::vector<double>, Eigen::VectorXd, Eigen::Array2d, Eigen::MatrixXd, Eigen::VectorXd> StateEstimator::reconstruct(
    py::EigenDRef<Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> image, 
    std::optional<AffineTrafo2D> new_trafo, Eigen::Array2i sitesShape, Eigen::Array2i projShape, int psfSupersample, 
    py::object &projector_generator, Eigen::Array2d phase_ref_image, Eigen::Array2d phase_ref_site,
    py::EigenDRef<Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> emissions, py::object &trafoSiteToImage)
{
    Eigen::Array2d outlier_ratios = this->imgPreproc->process_image(image);
    // Find trafo phase
    if(!new_trafo.has_value())
    {
        new_trafo.emplace(get_phase_shifted_trafo(image, projector_generator, phase_ref_image, phase_ref_site, trafoSiteToImage));
    }
    Eigen::VectorXd trafo_phase = get_phase_from_trafo_site_to_image(new_trafo.value(), phase_ref_image);
    // Construct local images
    std::vector<double> local_emissions = this->constructLocalImagesAndApplyProjectors(image, sitesShape, 
        new_trafo.value(), projShape, psfSupersample, projector_generator, emissions);
    return std::make_tuple(local_emissions, trafo_phase, outlier_ratios, new_trafo.value().matrix, new_trafo.value().offset);
}
