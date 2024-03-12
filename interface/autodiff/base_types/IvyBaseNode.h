#ifndef IVYBASENODE_H
#define IVYBASENODE_H


#include "autodiff/base_types/IvyThreadSafePtr.h"


/*
IvyBaseNode:
Base class of a node in the computation tree.
This is an empty class just so that we can detect the object to be a node by simply inheriting from this class.
*/
struct IvyBaseNode{};

using IvyBaseNodePtr_t = IvyThreadSafePtr_t<IvyBaseNode>;


#endif
