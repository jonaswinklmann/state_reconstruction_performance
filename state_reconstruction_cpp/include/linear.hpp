#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <vector>
#include <tuple>

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
    Eigen::MatrixXd matrix;
    Eigen::VectorXd offset;
    AffineTrafo() : 
        matrix(Eigen::Matrix2d()), offset(Eigen::Vector2d())
    {
        this->matrix << 1,0,0,1;
        this->offset << 0,0;
    };
    AffineTrafo(py::object& trafo);
    AffineTrafo(Eigen::MatrixXd matrix) : 
        matrix(matrix), offset(Eigen::Vector2d())
    {
        this->offset << 0,0;
    };
    AffineTrafo(Eigen::MatrixXd matrix, Eigen::VectorXd offset) : 
        matrix(matrix), offset(offset)
    {};
    AffineTrafo copy() const;
    Eigen::MatrixXd matrix_to_target() const;
    Eigen::MatrixXd matrix_to_origin() const;
    Eigen::VectorXd offset_to_target() const;
    Eigen::VectorXd offset_to_origin() const;
    unsigned int ndim() const;
    void set_offset_by_point_pair(Eigen::VectorXd origin_point, Eigen::VectorXd target_point);
    Eigen::VectorXd coord_to_origin(Eigen::VectorXd target_coords);
    Eigen::VectorXd coord_to_target(Eigen::VectorXd origin_coords);
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
    AffineTrafo2D(Eigen::MatrixXd matrix) : 
        AffineTrafo(matrix)
    {};
    AffineTrafo2D(Eigen::MatrixXd matrix, Eigen::VectorXd offset) : 
        AffineTrafo(matrix, offset)
    {};
    Eigen::Array2d getMagnification();
    Eigen::Array2d getAngle();
    Eigen::Array2d getOffset();
    void set_target_axes(std::optional<Eigen::Array2d> magnification,
        std::optional<Eigen::Array2d> angle, std::optional<Eigen::Array2d> offset);
    std::tuple<Eigen::Array2d, Eigen::Array2d, Eigen::Array2d> get_target_axes();
    void set_origin_axes(std::optional<Eigen::Array2d> magnification,
        std::optional<Eigen::Array2d> angle, std::optional<Eigen::Array2d> offset);
    std::tuple<Eigen::Array2d, Eigen::Array2d, Eigen::Array2d> get_origin_axes();
    std::tuple<std::vector<Eigen::Vector2d>, std::vector<Eigen::Vector2i>> filter_origin_coords_within_target_rect(
        const std::vector<Eigen::Vector2i>& coords, std::vector<Eigen::Array2i> rect);
};
