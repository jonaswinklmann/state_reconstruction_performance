#include <Eigen/Dense>
#include <optional>

class AffineTrafo2D;
Eigen::VectorXf get_phase_from_trafo_site_to_image(AffineTrafo2D trafo_site_to_image, std::optional<Eigen::VectorXf> phase_ref_image);