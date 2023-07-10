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
        matrixInfo.shape[0], matrixInfo.shape[1], strides).cast<float>();//trafo.attr("matrix").cast<Eigen::MatrixXf>();
    py::array offsetPy = trafo.attr("offset").cast<py::array>();
    this->offset = Eigen::Map<Eigen::VectorXd>(static_cast<Eigen::MatrixXd::Scalar*>(offsetPy.request().ptr), 
        offsetPy.size()).cast<float>();//trafo.attr("offset").cast<Eigen::VectorXf>();
}

AffineTrafo AffineTrafo::copy() const
{
    Eigen::MatrixXf m = this->matrix;
    Eigen::VectorXf o = this->offset;
    return AffineTrafo(std::move(m), std::move(o));
}

Eigen::MatrixXf AffineTrafo::matrix_to_target() const
{
    return this->matrix;
}

Eigen::MatrixXf AffineTrafo::matrix_to_origin() const
{
    return this->matrix.inverse();
}

Eigen::VectorXf AffineTrafo::offset_to_target() const
{
    return this->offset;
}

Eigen::VectorXf AffineTrafo::offset_to_origin() const
{
    return -this->matrix_to_origin() * this->offset;
}

unsigned int AffineTrafo::ndim() const
{
    return this->offset.size();
}

void AffineTrafo::set_offset_by_point_pair(Eigen::VectorXf origin_point, Eigen::VectorXf target_point)
{
    this->offset = target_point - this->matrix * origin_point;
}

Eigen::VectorXf AffineTrafo::coord_to_origin(Eigen::VectorXf target_coords)
{
    /*Transforms given target coordinates into origin coordinates.

    See :py:meth:`coord_to_target`.*/

    Eigen::MatrixXf mot = this->matrix_to_origin();
    Eigen::VectorXf bot = this->offset_to_origin();
    Eigen::VectorXf co = mot * target_coords + bot;
    return co;
}

Eigen::VectorXf AffineTrafo::coord_to_target(Eigen::VectorXf origin_coords)
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

    Eigen::MatrixXf mto = this->matrix_to_target();
    Eigen::VectorXf bto = this->offset_to_target();
    Eigen::VectorXf ct = mto * origin_coords + bto;
    return ct;
}

Eigen::Array2f AffineTrafo2D::getMagnification()
{
    Eigen::MatrixXf mot = this->matrix_to_origin();
    Eigen::Array2f res;

    res << sqrt(mot(0,0) * mot(0,0) + mot(1,0) * mot(1,0));
    res << sqrt(mot(0,1) * mot(0,1) + mot(1,1) * mot(1,1));
    return res;
}

Eigen::Array2f AffineTrafo2D::getAngle()
{
    Eigen::MatrixXf mot = this->matrix_to_origin();
    Eigen::Array2f res;
    res << atan2(mot(1,0), mot(0,0));
    res << atan2(-mot(0,1), mot(1,1));
    return res;
}

Eigen::Array2f AffineTrafo2D::getOffset()
{
    return this->offset_to_origin();
}