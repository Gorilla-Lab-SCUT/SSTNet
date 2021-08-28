#include "tree.h"


// build hierarchical tree according to the connection
Tree::Tree(DoubleList &c)
{
    connection = c;
    numLeaves = connection.size() + 1;
    amount = numLeaves + connection.size();
}

int Tree::getNum()
{ return amount; }

int Tree::getRoot()
{ return amount - 1; }

bool Tree::isLeaf(int id)
{
    if (id >= amount)
    { throw "id is out of range!"; }
    return (id < numLeaves);
}


NumList Tree::getLeaves(int id)
{
    NumList leaves = {};
    if (isLeaf(id)) {
        leaves.push_back(id);
    } else {
        // traverse child]
        int idx = id - numLeaves;
        NumList childrenIds = connection[idx]; // children's ids
        for (Int i = 0; i < 2; ++i) {
            Int childId = childrenIds[i];
            NumList childLeaves = getLeaves(childId); // get children's leaves
            leaves.insert(leaves.end(), childLeaves.begin(), childLeaves.end());
        }
    }
    return leaves;
}


std::tuple<NumList, NumList, NumList, NumList, NumList, NumList> Tree::fusionRecord()
{
    NumList leftList;
    NumList rightList;
    NumList fusionList;
    NumList leftIds;
    NumList rightIds;
    NumList fusionIds;
    for (Int i = 0; i < connection.size(); ++i) {
        NumList connect = connection[i];
        Int c0 = connect[0];
        Int c1 = connect[1];
        // get both leaves
        NumList leavesLeft = getLeaves(c0);
        NumList leavesRight = getLeaves(c1);
        // copy and concat
        NumList leavesFusion;
        leavesFusion = leavesLeft;
        // std::copy(leavesLeft.begin(), leavesLeft.end(), std::back_insert_iterator(leavesFusion));
        leavesFusion.insert(leavesFusion.end(), leavesRight.begin(), leavesRight.end());
        // concat to record (TODO wrap a function)
        leftList.insert(leftList.end(), leavesLeft.begin(), leavesLeft.end());
        rightList.insert(rightList.end(), leavesRight.begin(), leavesRight.end());
        fusionList.insert(fusionList.end(), leavesFusion.begin(), leavesFusion.end());
        NumList tempIdsLeaf (leavesLeft.size(), i); 
        NumList tempIdsRight (leavesRight.size(), i); 
        NumList tempIdsFusion (leavesFusion.size(), i); 
        leftIds.insert(leftIds.end(), tempIdsLeaf.begin(), tempIdsLeaf.end());
        rightIds.insert(rightIds.end(), tempIdsRight.begin(), tempIdsRight.end());
        fusionIds.insert(fusionIds.end(), tempIdsFusion.begin(), tempIdsFusion.end());
    }
    return std::tuple<NumList, NumList, NumList, NumList, NumList, NumList>(leftList, leftIds, rightList, rightIds, fusionList, fusionIds);
}

