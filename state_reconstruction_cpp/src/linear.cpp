#include "linear.hpp"

#include <cmath>
#include <pybind11/numpy.h>


constexpr bool rowMajor = Eigen::MatrixXd::Flags & Eigen::RowMajorBit;
typedef Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> StrideDyn;

AffineTrafo::AffineTrafo(py::object& trafo)
{
    py::array matrixPy = trafo.attr("matrix").cast<py::array>();
    auto matrixInfo = matrixPy.request();
    auto strides = StrideDyn(
        matrixInfo.strides[rowMajor ? 0 : 1] / (py::ssize_t)sizeof(Eigen::MatrixXd::Scalar),
        matrixInfo.strides[rowMajor ? 1 : 0] / (py::ssize_t)sizeof(Eigen::MatrixXd::Scalar));
    this->matrix = Eigen::Map<Eigen::MatrixXd, 0, StrideDyn>(static_cast<Eigen::MatrixXd::Scalar*>(matrixInfo.ptr), 
        matrixInfo.shape[0], matrixInfo.shape[1], strides); //trafo.attr("matrix").cast<Eigen::MatrixXd>();
    py::array offsetPy = trafo.attr("offset").cast<py::array>();
    this->offset = Eigen::Map<Eigen::VectorXd>(static_cast<Eigen::MatrixXd::Scalar*>(offsetPy.request().ptr), 
        offsetPy.size());//trafo.attr("offset").cast<Eigen::VectorXd>();
}

AffineTrafo AffineTrafo::copy() const
{
    Eigen::MatrixXd m = this->matrix;
    Eigen::VectorXd o = this->offset;
    return AffineTrafo(std::move(m), std::move(o));
}

Eigen::MatrixXd AffineTrafo::matrix_to_target() const
{
    return this->matrix;
}

Eigen::MatrixXd AffineTrafo::matrix_to_origin() const
{
    return this->matrix.inverse();
}

Eigen::VectorXd AffineTrafo::offset_to_target() const
{
    return this->offset;
}

Eigen::VectorXd AffineTrafo::offset_to_origin() const
{
    return -this->matrix_to_origin() * this->offset;
}

unsigned int AffineTrafo::ndim() const
{
    return this->offset.size();
}

void AffineTrafo::set_offset_by_point_pair(Eigen::VectorXd origin_point, Eigen::VectorXd target_point)
{
    this->offset = target_point - this->matrix * origin_point;
}

Eigen::VectorXd AffineTrafo::coord_to_origin(Eigen::VectorXd target_coords)
{
    /*Transforms given target coordinates into origin coordinates.

    See :py:meth:`coord_to_target`.*/

    Eigen::MatrixXd mot = this->matrix_to_origin();
    Eigen::VectorXd bot = this->offset_to_origin();
    Eigen::VectorXd co = mot * target_coords + bot;
    return co;
}

Eigen::VectorXd AffineTrafo::coord_to_target(Eigen::VectorXd origin_coords)
{
    /*Transforms given origin coordinates into target coordinates.

    Parameters
    ----------
    origin_coords : `np.ndarray(float)`
        Coordinates in origin space. The different dimensions should be
        placed on the last axes (dimensions: [..., ndim]).

    Returns
    -------
    target_coords : `np.ndarray(float)`
        Transformed coordinates in target space.*/

    Eigen::MatrixXd mto = this->matrix_to_target();
    Eigen::VectorXd bto = this->offset_to_target();
    Eigen::VectorXd ct = mto * origin_coords + bto;
    return ct;
}

Eigen::Array2d AffineTrafo2D::getMagnification()
{
    Eigen::MatrixXd mot = this->matrix_to_origin();
    Eigen::Array2d res;

    res << sqrt(mot(0,0) * mot(0,0) + mot(1,0) * mot(1,0));
    res << sqrt(mot(0,1) * mot(0,1) + mot(1,1) * mot(1,1));
    return res;
}

Eigen::Array2d AffineTrafo2D::getAngle()
{
    Eigen::MatrixXd mot = this->matrix_to_origin();
    Eigen::Array2d res;
    res << atan2(mot(1,0), mot(0,0));
    res << atan2(-mot(0,1), mot(1,1));
    return res;
}

Eigen::Array2d AffineTrafo2D::getOffset()
{
    return this->offset_to_origin();
}

std::tuple<std::vector<Eigen::Vector2d>, std::vector<Eigen::Vector2i>> AffineTrafo2D::filter_origin_coords_within_target_rect(
    const std::vector<Eigen::Vector2i> &coords, std::vector<Eigen::Array2i> rect)
{
    std::vector<Eigen::Vector2d> emissionCoords;
    std::vector<Eigen::Vector2i> originCoords;
    for(const auto& coord : coords)
    {
        auto shiftedCoords = coord_to_target(coord.cast<double>());
        if(shiftedCoords[0] >= rect[0][0] && shiftedCoords[0] <= rect[0][1] && 
            shiftedCoords[1] >= rect[1][0] && shiftedCoords[1] <= rect[1][1])
            {
                emissionCoords.push_back(shiftedCoords);
                originCoords.push_back(coord);
            }
    }
    return std::make_tuple(emissionCoords, originCoords);
}
