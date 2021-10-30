
#ifndef PRIORITYQUEUUE_H // include guard
#define PRIORITYQUEUUE_H
#include <math.h> 
#include <iostream>
#include <vector>
#include <algorithm>
#include <stdexcept>
using namespace std;
class PriorityQueue
{
public:
    PriorityQueue() {};
    ~PriorityQueue() {};
    // return size of the heap
    unsigned int size();

    // Function to check if the heap is empty or not
    bool empty();

    // insert key into the heap
    void push(int key);

    // Function to remove an element with the lowest priority (present at the root)
    void pop();

    // Function to return an element with the lowest priority (present at the root)
    int top();

private:
    // vector to store heap elements
    vector<int> A;

    const int ROOT = 0;
    // return parent of `A[i]`
    // don't call this function if `i` is already a root node
    int PARENT(int i);

    // return left child of `A[i]`
    int LEFT(int i);

    // return right child of `A[i]`
    int RIGHT(int i);

    // Recursive heapify-down algorithm.
    // The node at index `i` and its two direct children
    // violates the heap property
    void heapify_down(int i);

    // Recursive heapify-up algorithm
    // Heapify up is used when the parent of an element.
    // violates the heap property. It converts the binary
    // tree into a heap by moving the element up the the tree.
    void heapify_up(int i);

};
#endif /* PRIORITYQUEUUE_H */