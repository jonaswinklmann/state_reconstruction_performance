#include <Eigen/Dense>
#include <pybind11/pybind11.h>

namespace py = pybind11;

class AffineTrafo
{
    /*Defines an affine transformation in arbitrary dimensions.

    Parameters
    ----------
    matrix : `np.ndarray(2, float)`
        Transformation matrix.
    offset : `np.ndarray(1, float)`
        Transformation offset.*/
public:
    Eigen::MatrixXf matrix;
    Eigen::VectorXf offset;
    AffineTrafo() : 
        matrix(Eigen::Matrix2f()), offset(Eigen::Vector2f())
    {
        this->matrix << 1,0,0,1;
        this->offset << 0,0;
    };
    AffineTrafo(py::object& trafo);
    AffineTrafo(Eigen::MatrixXf matrix) : 
        matrix(matrix), offset(Eigen::Vector2f())
    {
        this->offset << 0,0;
    };
    AffineTrafo(Eigen::MatrixXf matrix, Eigen::VectorXf offset) : 
        matrix(matrix), offset(offset)
    {};
    AffineTrafo copy() const;
    Eigen::MatrixXf matrix_to_target() const;
    Eigen::MatrixXf matrix_to_origin() const;
    Eigen::VectorXf offset_to_target() const;
    Eigen::VectorXf offset_to_origin() const;
    unsigned int ndim() const;
    void set_offset_by_point_pair(Eigen::VectorXf origin_point, Eigen::VectorXf target_point);
    Eigen::VectorXf coord_to_origin(Eigen::VectorXf target_coords);
    Eigen::VectorXf coord_to_target(Eigen::VectorXf origin_coords);
};

class AffineTrafo2D : public AffineTrafo
{
    /*Maps origin pixel positions to target pixels positions.

    Convention: ``target_coord = matrix * origin_coord + offset``.

    Usage:

    * Take images for different single-pixel illuminations.
    * Calculate transformation parameters with :py:meth:`calc_trafo`.
    * Perform transforms with call method
      (or :py:meth:`cv_to_target`, :py:meth:`cv_to_origin`).*/
public:
    AffineTrafo2D() : 
        AffineTrafo()
    {};
    AffineTrafo2D(py::object& trafo) : 
        AffineTrafo(trafo)
    {};
    AffineTrafo2D(Eigen::MatrixXf matrix) : 
        AffineTrafo(matrix)
    {};
    AffineTrafo2D(Eigen::MatrixXf matrix, Eigen::VectorXf offset) : 
        AffineTrafo(matrix, offset)
    {};
    Eigen::Array2f getMagnification();
    Eigen::Array2f getAngle();
    Eigen::Array2f getOffset();
};
