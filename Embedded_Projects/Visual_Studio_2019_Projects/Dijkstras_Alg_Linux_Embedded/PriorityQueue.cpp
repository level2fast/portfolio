// C_plus_plus_Sandbox.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
/******************************************************************************

Explanation of Min and Max heaps found here:
https://www.techiedelight.com/introduction-priority-queues-using-binary-heaps/
Implement a heap data structure in c++
 Output:

Size is 3
2 3
Size is 4
4 5 15 45
true
Vector::at() : index is out of range(Heap underflow)
Vector::at() : index is out of range(Heap underflow)
*******************************************************************************/
#include <stdio.h>      /* printf */
#include "PriorityQueue.h"


// return parent of `A[i]`
// don't call this function if `i` is already a root node
int PriorityQueue::PARENT(int i)
{
    return floor((i - 1) / 2);
}

// return left child of `A[i]`
int PriorityQueue::LEFT(int i)
{
    return (2 * i + 1);
}

// return right child of `A[i]`
int PriorityQueue::RIGHT(int i)
{
    return (2 * i + 2);
}

// Recursive heapify-down algorithm.
// The node at index `i` and its two direct children
// violates the heap property
void PriorityQueue::heapify_down(int i)
{
    /* When removing an element, always remove the root
    1.Grab the root
    2.Swap the element at index into the roots position
    3.decrement index
    4.while the "out of place element" has a larger priority than any child, swap the out of place element with the smallest child
    this last step is known as "heapify down", and can be implemented recursively */

    //1.Grab the root
    int smallest = i;
    int left = LEFT(i);
    int right = RIGHT(i);

    // 2.Swap the element at index into the roots position
    // if current root is greater than the left child and
    // child is not outside of array bounds
    if (left < size() && A[smallest] > A[left])
    {
        //replace current root with left child 
        smallest = left;
    }

    // if current root is greater than the right child and
    // child is not outside of array bounds        
    if (right < size() && A[smallest] > A[right])
    {
        smallest = right;
    }

    // check if smaller node was found.
    if (smallest != i)
    {
        // Smaller node found so swap current root with node pointed to by 
        // smallest index and make recursive call
        swap(A[i], A[smallest]);
        heapify_down(i);
    }

}

// Recursive heapify-up algorithm
// Heapify up is used when the parent of an element.
// violates the heap property. It converts the binary
// tree into a heap by moving the element up the the tree.
void PriorityQueue::heapify_up(int i)
{
    // first compare the element with its parent and swap
    // the two if the heap property is violated.
    // We are going to move up the tree if the current element is 
    // smaller than its parent
    if (i != 0 && A[i] < A[PARENT(i)])
    {
        int temp = PARENT(i);

        //swap values of parent and child since parent is bigger
        swap(A[i], A[temp]);

        //call heapify up using parent index
        heapify_up(temp);
    }


}

// return size of the heap
unsigned int PriorityQueue::size()
{
    return A.size();
}

// Function to check if the heap is empty or not
bool PriorityQueue::empty()
{
    return size() == 0;
}

// insert key into the heap
void PriorityQueue::push(int key)
{
    //place element at the end of the array
    A.push_back(key);

    // call heapify up to ensure element is placed
    // in correct position in binary tree
    int temp = size() - 1;
    heapify_up(temp);
}

// Function to remove an element with the lowest priority (present at the root)
void PriorityQueue::pop()
{
    try
    {
        // if the heap has no elements, throw an exception
        if (size() == 0)
        {
            throw out_of_range("Vector<X>::at() : "
                "index is out of range(Heap underflow)");
        }

        // replace the root element with the last element
        // in the tree
        A[ROOT] = A[size() - 1];

        // remove last element in tree
        A.pop_back();

        // call heapify down startin with root to 
        // restructure tree 
        heapify_down(ROOT);
    }
    // catch and print the exception
    catch (const out_of_range& oor)
    {
        cout << "\n" << oor.what();
    }
}

// Function to return an element with the lowest priority (present at the root)
int PriorityQueue::top()
{
    try
    {
        // if the queue is empty, throw an exception
        if (size() == 0)
        {
            throw out_of_range("Vector<X>::at() : "
                "index is out of range(Heap underflow)");
        }
        // return the root of tree which always has the lowest priority
        return A[ROOT];
    }
    // catch and print the exception
    catch (const out_of_range& oor)
    {
        cout << "\n" << oor.what();
        return -1;
    }

}






// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
