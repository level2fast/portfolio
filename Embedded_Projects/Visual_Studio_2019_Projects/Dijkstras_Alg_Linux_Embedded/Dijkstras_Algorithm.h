#pragma once
#ifndef DIJKSTRAS_ALGORITHM_H // include guard
#define DIJKSTRAS_ALGORITHM_H
#include <queue>          // std::priority_queue
#include <map>
#include <vector>
#include <iostream>
using namespace std;
struct Node
{
    int id;
    vector<Node> neighbors; // list of neighbor nodes
    int edge;
    // Map to hold the weight of the shortest
    // path for each node we've visited
   static map<int, int>path_weight;
};
class compareWeights
{

public:

    bool operator()(const Node& lhs, const Node& rhs) const
    {
        if (lhs.path_weight[lhs.id] > rhs.path_weight[rhs.id])
        {
            return true;
        }
        else
        {
            return false;
        }
    }
};

class Graph
{
public:

    // A utility function that creates 
    // a graph of V vertices
    Graph(int V);
    // Adds and edge to the graph to connect
    // one node to another.
    void addEdge(Node *temp, int dest, int weight);
    vector<Node> nodes;
    // Prints entire tree
    void printTree();
};

class Dijkstras
{
public:
    // Map to hold the weight of the shortest
    // path for each node we've visited
    static map<int, int>path_weight;

    // Priority queue will hold remainig nodes
    // that we will visit during the shortest
    // path search
    priority_queue<Node, vector<Node>, compareWeights> myPQ;

    // Map to keep track of which node precedes
    // each node we visit. This will assist in
    // retracing our steps to determine the shortest
    // path.
    map<int, int>previousNode;

    // Fuction to find the shortest path 
    // starting from any node in the tree
    void findShortestPath(Node * root);

    // Prints each node and total distance to
    // get there from stat node
    void printShortestPath(int nodeId);

    void print(int key);



};
#endif //DIIJKSTRAS_ALGORITHM_H