/*main header file if you need it*/
/*the header file for main.cpp*/

#include <iostream>
#include <cmath>
#include <iomanip>
#include <string>
#include <sstream>
#include <fstream>

int main()
{
    ifstream myFile;
    myFile.open("input.txt");
    string line;
    if (myFile.is_open())
    {
        while(getline(myFile,line))
        {
            cout << line << "\n";
        }
    }
    return 0;
}