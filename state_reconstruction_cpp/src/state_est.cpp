#include <pybind11/numpy.h>
#include <optional>

#include "trafo_est.hpp"

class StateEstimator
{
public:
    int reconstruct(py::array_t<float, py::array::c_style | py::array::forcecast> image) {
        
        return 1;
    }
    /*void get_phase_shifted_trafo(std::optional<py::tuple> phase=std::nullopt, 
        std::optional<py::array_t<float, py::array::c_style | py::array::forcecast>> image = std::nullopt, 
        std::string method="projection_optimization", bool preprocess_image=true)
    {
        Gets a phase-shifted lattice transformation from an image.

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
            Phase-shifted lattice transformation.

        if (!phase.has_value())
        {
            // Image preprocessing
            if (!phase.has_value())
            {
                throw std::runtime_error("No `image` or `phase` given");
            }
            if (preprocess_image && self.image_preprocessor)
            {
                image, _ = self.image_preprocessor.process_image(image)
            }
            // From isolated atoms
            if (method == "isolated_atoms" && self.isolated_locator is not None)
            {
                label_centers = self.isolated_locator.get_label_centers(image)
                if len(label_centers) > 0:
                    phase, _ = get_trafo_phase_from_points(
                        *np.moveaxis(label_centers, -1, 0),
                        self.trafo_site_to_image
                    )
                else:
                    phase = np.zeros(2)
                    self.LOGGER.error(
                        f"get_phase_shifted_trafo: "
                        f"Could not locate isolated atoms. "
                        f"Using default phase: {str(phase)}"
                    )
            }
            // From emission variance maximization
            else
            {
                if (method != "projection_optimization")
                {
                    self.LOGGER.warning(
                        f"get_phase_shifted_trafo: "
                        f"Method `{str(method)}` unavailable, "
                        f"using `projection_optimization`"
                    )
                }
                phase = get_trafo_phase_from_projections(
                    image, self.projector_generator,
                    phase_ref_image=self.phase_ref_image,
                    phase_ref_site=self.phase_ref_site
                )
            }
        }
        // Construct phase-shifted trafo
        new_trafo = trafo_gen.get_trafo_site_to_image(
            trafo_site_to_image=self.trafo_site_to_image, phase=phase,
            phase_ref_site=self.phase_ref_site,
            phase_ref_image=self.phase_ref_image
        )
        return new_trafo
    }*/
};