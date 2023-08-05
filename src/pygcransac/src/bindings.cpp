#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include "gcransac_python.h"


namespace py = pybind11;

py::tuple findRigidTransform(
	py::array_t<double>  x1y1z1x2y2z2_,
	py::array_t<double>  probabilities_,
	double threshold,
	double conf,
	double spatial_coherence_weight,
	int max_iters,
	int min_iters,
	bool use_sprt,
	double min_inlier_ratio_for_sprt,
	int sampler,
	int neighborhood,
	int lo_number,
	double neighborhood_size,
	bool use_space_partitioning,
	double sampler_variance)
{
	py::buffer_info buf1 = x1y1z1x2y2z2_.request();
	size_t NUM_TENTS = buf1.shape[0];
	size_t DIM = buf1.shape[1];

	if (DIM != 6) {
		throw std::invalid_argument("x1y1z1x2y2z2 should be an array with dims [n,6], n>=3");
	}
	if (NUM_TENTS < 3) {
		throw std::invalid_argument("x1y1z1x2y2z2 should be an array with dims [n,6], n>=3");
	}

	double *ptr1 = (double *)buf1.ptr;
	std::vector<double> x1y1z1x2y2z2;
	x1y1z1x2y2z2.assign(ptr1, ptr1 + buf1.size);

    std::vector<double> probabilities;
    if (sampler == 3 || sampler == 4)
    {
        py::buffer_info buf_prob = probabilities_.request();
        double* ptr_prob = (double*)buf_prob.ptr;
        probabilities.assign(ptr_prob, ptr_prob + buf_prob.size);        
    }

	std::vector<double> pose(16);
	std::vector<bool> inliers(NUM_TENTS);

	int num_inl = findRigidTransform_(
		x1y1z1x2y2z2,
		probabilities,
		inliers,
		pose,
		spatial_coherence_weight,
		threshold,
		conf,
		max_iters,
		min_iters,
		use_sprt,
		min_inlier_ratio_for_sprt,
		sampler,
		neighborhood,
		neighborhood_size,
		sampler_variance,
		use_space_partitioning,
		lo_number);

	py::array_t<bool> inliers_ = py::array_t<bool>(NUM_TENTS);
	py::buffer_info buf3 = inliers_.request();
	bool *ptr3 = (bool *)buf3.ptr;
	for (size_t i = 0; i < NUM_TENTS; i++)
		ptr3[i] = inliers[i];
	if (num_inl == 0) {
		return py::make_tuple(pybind11::cast<pybind11::none>(Py_None), inliers_);
	}
	py::array_t<double> pose_ = py::array_t<double>({ 4,4 });
	py::buffer_info buf2 = pose_.request();
	double *ptr2 = (double *)buf2.ptr;
	for (size_t i = 0; i < 16; i++)
		ptr2[i] = pose[i];
	return py::make_tuple(pose_, inliers_);
}



PYBIND11_PLUGIN(pygcransac) {

    py::module m("pygcransac", R"doc(
        Python module
        -----------------------
        .. currentmodule:: pygcransac
        .. autosummary::
           :toctree: _generate

		   findRigidTransform,

    )doc");

	m.def("findRigidTransform", &findRigidTransform, R"doc(some doc)doc",
		py::arg("x1y1z1x2y2z2"),
        py::arg("probabilities"),
		py::arg("threshold") = 1.0,
		py::arg("conf") = 0.99,
		py::arg("spatial_coherence_weight") = 0.975,
		py::arg("max_iters") = 10000,
		py::arg("min_iters") = 50,
		py::arg("use_sprt") = false,
		py::arg("min_inlier_ratio_for_sprt") = 0.00001,
		py::arg("sampler") = 1,
		py::arg("neighborhood") = 1,
		py::arg("lo_number") = 50,
		py::arg("neighborhood_size") = 2.0,
		py::arg("use_space_partitioning") = true,
		py::arg("sampler_variance") = 0.1);

  return m.ptr();
}
