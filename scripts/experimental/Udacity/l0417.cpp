/*Write a program that asks a user for five numbers.
**Print out the sum and average of the five numbers.
*/

#include <iostream>

int main()
{
    float i_1, sum, ave = 0.0;
    std::cout<<"Enter 5 numbers, we'll compute the avg; first number:\n";
    for (int i = 0; i < 5; i++)
    {
        std::cin>>i_1;
        sum = sum + i_1;
    }
    std::cout<<"Computing...\n";
    ave = sum/5.0;
    std::cout<<"Sum: "<<sum<<"\n";
    std::cout<<"Average: "<<ave<<"\n";
    return 0;
}