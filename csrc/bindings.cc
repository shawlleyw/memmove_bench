#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cpp_move.hh"
#include "cuda_move.h"

namespace py = pybind11;

PYBIND11_MODULE(memmove_c, m) {
    m.def("permute_tokens_cpp", &permute_tokens_cpp);
    m.def("permute_tokens_cuda", &permute_tokens_cuda);
};