#include <cmath>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <exception>
#include <iostream>
#include <fstream>

#include "linear.hpp"
#include "optimize.hpp"
#include "proj_est.hpp"
#include "trafo_est.hpp"
#include "trafo_gen.hpp"
#include "calcEmissions.hpp"

AffineTrafo2D get_shifted_subimage_trafo(AffineTrafo2D trafo, Eigen::VectorXd shift, 
    Eigen::VectorXd subimage_center, std::optional<Eigen::VectorXd> site = std::nullopt)
{
    // Gets the lattice transformation from subimage center and shift.
    if(!site.has_value())
    {
        site = Eigen::Vector2d();
        site.value() << 0,0;
    }
    AffineTrafo2D shifted_trafo = AffineTrafo2D(trafo.matrix, trafo.offset);
    shifted_trafo.set_offset_by_point_pair(site.value(), subimage_center + shift);
    return shifted_trafo;
}

double get_subimage_emission_std(Eigen::VectorXd shift, Eigen::Array2i subimage_center, 
    const Eigen::Array<double,-1,-1,Eigen::RowMajor>& full_image, py::object& prjgen, Eigen::Array2i subsite_shape = {5, 5})
{
    // Performs projection with given transformation shift and subimage.
    // Parse parameters
    // tmp_prjgen = copy.deepcopy(prjgen)
    py::object prjgenTrafo = prjgen.attr("trafo_site_to_image");
    AffineTrafo2D prjgenTrafoCpp(prjgenTrafo);
    Eigen::Array2i projShape = prjgen.attr("proj_shape").cast<Eigen::Array2i>();
    int psfSupersample = prjgen.attr("psf_supersample").cast<int>();
    
    // Set sites
    std::vector<Eigen::Array2i> subsiteCoords;
    for(int i = (int)(floor((double)(-subsite_shape[0]) / 2.)); i < (subsite_shape[0] + 1) / 2 + 1; i++)
    {
        for(int j = (int)(floor((double)(-subsite_shape[1]) / 2.)); j < (subsite_shape[1] + 1) / 2 + 1; j++)
        {
            Eigen::Array2i coords;
            coords << i,j;
            subsiteCoords.push_back(std::move(coords));
        }
    }
    // Find coordinates and keep only sites within image
    AffineTrafo2D shiftedTrafo = get_shifted_subimage_trafo(prjgenTrafoCpp, shift, subimage_center.cast<double>());
    Eigen::Array2i halfProjShape = projShape / 2;
    std::vector<Eigen::Vector2d> subimageCoords;
    for (const Eigen::Array2i& coord : subsiteCoords)
    {
        Eigen::Vector2d shiftedCoord = shiftedTrafo.coord_to_target(coord.cast<double>());
        if(shiftedCoord[0] > (halfProjShape[0] + 1) && shiftedCoord[0] < (full_image.rows() - halfProjShape[0] - 1) && 
            shiftedCoord[1] > (halfProjShape[1] + 1) && shiftedCoord[1] < (full_image.cols() - halfProjShape[1] - 1))
        {
            subimageCoords.push_back(shiftedCoord);
        }
    }
    // Find local images
    // tmp_prjgen.trafo_site_to_image = _trafo
    auto localImages = getLocalImages(subimageCoords, full_image, projShape, psfSupersample);
    // Perform projection
#ifdef CUDA
    auto emissions = EmissionCalculatorCUDA::getInstance().calcEmissionsGPU(localImages, psfSupersample);
#else
    auto emissions = apply_projectors(localImages, prjgen);
#endif
    Eigen::ArrayXd emissionsEigen(Eigen::Map<Eigen::ArrayXd>(emissions.data(), emissions.size()));
    return std::sqrt((emissionsEigen - emissionsEigen.mean()).square().sum() / (emissionsEigen.size()));
}

std::vector<int> getSubsiteShape(py::object& prjgen, Eigen::Array2i& subimageShape, const std::vector<int> minShape={3, 3})
{
    // Gets the default subimage sites shape.
    py::object affineTrafo = prjgen.attr("trafo_site_to_image");
    py::object getOriginAxesFunction = affineTrafo.attr("get_origin_axes");
    std::vector<double> magnification = getOriginAxesFunction().cast<std::vector<std::vector<double>>>()[0];
    Eigen::Array2i subsiteSize(subimageShape.cast<double>().mean() * 2 / (magnification[0] + magnification[1]) + 0.5);
    std::vector<int> ret(minShape);
    if(subsiteSize[0] > ret[0])
    {
        ret[0] = subsiteSize[0];
    }
    if(subsiteSize[1] > ret[1])
    {
        ret[1] = subsiteSize[1];
    }
    return ret;
}

Eigen::VectorXd TrafoEstimator::get_trafo_phase_from_projections(const py::EigenDRef<const Eigen::Array<double,-1,-1,Eigen::RowMajor>>& im, 
    py::object& prjgen, Eigen::Array2d phase_ref_image, Eigen::Array2d phase_ref_site,
    std::optional<std::vector<int>> subimage_shape, std::optional<std::vector<int>> subsite_shape, int search_range)
{
    /*Gets the lattice phase by maximizing the emission standard deviation.

    Parameters
    ----------
    im : `Array[2, float]`
        Fluorescence image.
    prjgen : `srec.ProjectionGenerator`
        Projection generator object.
    phase_ref_image, phase_ref_site : `(int, int)`
        Lattice phase reference in fluorescence image coordinates.
    subimage_shape : `(int, int)` or `None`
        Shape of subimages (subdivisions of full image) used for
        standard deviation evaluation.
    subsite_shape : `(int, int)` or `None`
        Shape of sites used for projection.
    search_range : `int`
        Discrete optimization search range.
        See :py:func:`libics.tools.math.optimize.minimize_discrete_stepwise`.

    Returns
    -------
    phase : `np.ndarray(1, float)`
        Phase (residual) of lattice w.r.t. to image coordinates (0, 0).*/

    // Parse parameters
    if (!subimage_shape.has_value())
    {
        subimage_shape = prjgen.attr("psf_shape").cast<std::vector<int>>();
    }
    Eigen::Array2i subimageShapeEigen(subimage_shape.value().data());
    if (subimageShapeEigen[0] == 0 || subimageShapeEigen[1] == 0)
    {
        throw std::invalid_argument("SubimageShapeEigen must not be 0");
    }
    // Cropped image to avoid checking image borders
    Eigen::Array2i projShape = prjgen.attr("proj_shape").cast<Eigen::Array2i>();

    if (projShape[0] > im.rows() || im.rows() < 2 * projShape[0] || projShape[1] > im.cols() || im.cols() < 2 * projShape[1])
    {
        throw std::invalid_argument("ProjShape not appropriate");
    }
    Eigen::Array<double,-1,-1,Eigen::RowMajor> imRoi = im(Eigen::seq(projShape[0], Eigen::last - projShape[0]), Eigen::seq(projShape[1], Eigen::last - projShape[1]));
    Eigen::Array2i cropShape = (Eigen::Array2i({imRoi.rows(), imRoi.cols()}) / subimageShapeEigen).cast<int>() * subimageShapeEigen;

    if (cropShape[0] > imRoi.rows() || cropShape[1] > imRoi.cols())
    {
        throw std::invalid_argument("CropShape not appropriate");
    }
    Eigen::Array<double,-1,-1,Eigen::RowMajor> imCrop = imRoi(Eigen::seqN(0, cropShape[0]), Eigen::seqN(0, cropShape[1]));
    Eigen::Array2i gridShape = cropShape / subimageShapeEigen;

    // If GPU used, init environment and load data
#ifdef CUDA
    EmissionCalculatorCUDA& emissionCalcGPU = EmissionCalculatorCUDA::getInstance();
    emissionCalcGPU.initGPUEnvironment();
    emissionCalcGPU.loadImage(im.data(), im.cols(), im.rows());
    //emissionCalcGPU.loadProj(prjgen);
#endif

    // Get subimage with maximum signal variance as proxy for mixed filling
    int iMax = -1;
    int jMax = -1;
    double stdevMax = -1;
    for(int i = 0; i < gridShape[0]; i++)
    {
        for(int j = 0; j < gridShape[1]; j++)
        {
            Eigen::Array<double,-1,-1,Eigen::RowMajor> subImage = imCrop(Eigen::seqN(i * subimageShapeEigen[0], 
                subimageShapeEigen[0]), Eigen::seqN(j * subimageShapeEigen[1], subimageShapeEigen[1]));
            double stdev = std::sqrt((subImage.array() - subImage.mean()).square().sum() / subImage.size());
            if(stdev > stdevMax)
            {
                stdevMax = stdev;
                iMax = i;
                jMax = j;
            }
        }
    }
    Eigen::Array2i subimageCenter = (Eigen::Array2d({iMax + 0.5, jMax + 0.5}) * (subimageShapeEigen.cast<double>())).cast<int>() + projShape;

    // Get phase by maximizing projected emissions variance
    if (!subsite_shape.has_value())
    {
        subsite_shape = getSubsiteShape(prjgen, subimageShapeEigen);
    }
    Eigen::Array2i subsiteShapeEigen(subsite_shape.value().data());
    Eigen::VectorXd initShift(2);
    initShift << 0,0;
    Eigen::VectorXd dx(1);
    dx << 1;

#ifdef CUDA
    int imageCount = ((int)((subsiteShapeEigen[0] + 1) / 2) * 2 + 1) * ((int)((subsiteShapeEigen[1] + 1) / 2) * 2 + 1);
    emissionCalcGPU.allocatePerImageBuffers(imageCount);
#endif

    // Maximize on integer pixels
    std::map<Eigen::VectorXd,double,EigenVectorXdCompare> resultsCache = std::map<Eigen::VectorXd,double,EigenVectorXdCompare>();
    Eigen::VectorXi optShiftInt = maximize_discrete_stepwise_cpp(get_subimage_emission_std, initShift, 
        resultsCache, dx, search_range, 10000, subimageCenter, im, prjgen, subsiteShapeEigen).cast<int>();

    // Maximize on subpixels
    double psfSuperSample = prjgen.attr("psf_supersample").cast<double>();
    dx = Eigen::VectorXd(1);
    dx << 1. / psfSuperSample;
    Eigen::VectorXd optShiftFloat = maximize_discrete_stepwise_cpp(get_subimage_emission_std, optShiftInt.cast<double>(),
        resultsCache, dx, (int)(ceil(search_range * psfSuperSample / 2)), 10000, subimageCenter, im, prjgen, subsiteShapeEigen);
        
    // Calculate phase
    py::object trafo = prjgen.attr("trafo_site_to_image");
    AffineTrafo2D optTrafo = get_shifted_subimage_trafo(AffineTrafo2D(trafo), optShiftFloat, subimageCenter.cast<double>(), phase_ref_site);
    Eigen::VectorXd phase = get_phase_from_trafo_site_to_image(optTrafo, phase_ref_image);
    return phase;
}