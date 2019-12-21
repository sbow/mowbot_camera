/*Student Playground*/
/*Put your code here*/

#include <iostream>
#include <cmath>
#include <iomanip>
#include <string>
#include <sstream>
#include <fstream>

int main()
{
    float num, sum, mean = 0;
    int n = 0;
    
    std::ifstream myFile;
    myFile.open("input.txt");
    std::string line;
    if (myFile.is_open())
    {
        while(getline(myFile,line))
        {
            std::cout << line << "\n";      // read line
            std::stringstream(line) >> num; // convert to float
            sum += num;                     // add to sum
            n += 1;                         // increment number of elements
        }
    }
    mean = sum/n;
    std::cout << "Sum: "<<std::setw(15)<<" Mean: "<<std::setw(15)<<"\n";
    std::cout << sum    <<std::setw(15)<< mean    <<std::setw(15)<<"\n";
    return 0;
}