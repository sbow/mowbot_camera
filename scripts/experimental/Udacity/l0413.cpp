/*Goal: understand the switch statement in C++
**This example does not use a break statement between 
**the possibilities, which means all menu items below the selected
**one are executed. 
**
**Note: even if L is selected, S & B run too, ie: it's like a stacking option
**sheet. Only at the end of B, does the break get executed, keeping default
**from being executed.
*/


#include<iostream>

int main()
{
    char menuItem;
    std::cout<<"Choose your holiday package:\n";
    std::cout<<"L: luxury package\nS: standard package\n";
    std::cout<<"B: basic package ";
    
    std::cin>>menuItem;
    std::cout<<menuItem<<"\n";
        std::cout<<"The "<<menuItem<<" package includes:\n";
    
    switch(menuItem)
    {
        case 'L':  // note: need to use single quotes for char compare
        {
            std::cout<<"\tSpa Day\n";
            std::cout<<"\tSailboat Tour\n";
        }
        case 'S':
        {
            std::cout<<"\tCity Tour\n";
            std::cout<<"\tComplimentary Happy Hour\n";  
        }
        case 'B':
        {
            std::cout<<"\tAirport Transfers\n";
            std::cout<<"\tComplimentary Breakfast\n"; 
            break;
        }
        default:
            std::cout<<"Please select the L,S,B package.\n";
    }
    return 0;
}

