#ifndef TREE_H
#define TREE_H
#include <vector>
#include <boost/range/irange.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/embed.h> // everything needed for embedding
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
#include <pybind11/complex.h>

using Int = int32_t;
using NumList = std::vector<Int>;
using DoubleList = std::vector<NumList>;

class Tree
{
private:
    DoubleList connection;
    int numLeaves;
    int amount;
public:
    Tree(DoubleList &c);
    int getNum();
    int getRoot();
    bool isLeaf(int id);
    NumList getLeaves(int id);
    std::tuple<NumList, NumList, NumList, NumList, NumList, NumList> fusionRecord();
};

#endif

