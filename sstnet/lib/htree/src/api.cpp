#include "tree.h"
// #include <torch/extension.h>

namespace py = pybind11;

#define TORCH_EXTENSION_NAME htree

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.doc() = "hierarchy construction";
    
    py::class_<Tree>(m, "Tree")
        .def(py::init<DoubleList&>())
        .def("num", &Tree::getNum, "get the number of nodes")
        .def("root", &Tree::getRoot, "get the root id")
        .def("is_leaf", &Tree::isLeaf, "judge whether is leaf or not")
        .def("get_leaves", &Tree::getLeaves, "get leaves according to given id")
        .def("fusion_record", &Tree::fusionRecord, "fusion and record the process");
}

