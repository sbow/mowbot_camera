/*Goal: Examine pointers!*/
/* clang++ -std=c++11 -O3 -o l0503.o l0503pointers.cpp */

#include <iostream>

int main()
{
    int a = 54;
    int * ap = &a; // &a is address of a, int * ap is a pointer to an int datatype
    std::cout<<"a = "<<a<<"\n";
    std::cout<<"address of a is at &a = "<< &a<<"\n";
    std::cout<<"value at &a= "<<* ap << "\n";
    return 0;
}