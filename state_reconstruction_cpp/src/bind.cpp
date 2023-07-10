#include <pybind11/stl.h>

#include "linear.hpp"
#include "optimize.hpp"
#include "trafo_est.hpp"

PYBIND11_MODULE(state_reconstruction_cpp, m) {
    m.doc() = "pybind11 state_reconstruction_cpp module";

    py::class_<TrafoEstimator>(m, "TrafoEstimator")
        .def(py::init<>())
        .def("get_trafo_phase_from_projections", &TrafoEstimator::get_trafo_phase_from_projections, py::arg("im"), py::arg("prjgen"), 
        py::arg("phase_ref_image")=py::make_tuple(0, 0), py::arg("phase_ref_site")=py::make_tuple(0, 0), 
        py::arg("subimage_shape")=std::nullopt, py::arg("subsite_shape")=std::nullopt, py::arg("search_range")=1);

    /*py::class_<StateEstimator>(m, "StateEstimator")
        .def(py::init<>())
        .def("reconstruct", &StateEstimator::reconstruct, "A function that reconstructs an image", py::arg("image"));*/
}