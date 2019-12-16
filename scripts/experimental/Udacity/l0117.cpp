/* l107.cpp
*/
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>

using namespace std;

enum MONTHS {Jan, Feb, Mar};

int main()
{
    const int my_int_ = 4543;
    string line;
    MONTHS my_month_ = Jan;
    if (my_month_ == Jan)
    {
        cout << "hello world! Val of int: "<<my_int_<<"\n";
        cout << "Names"<<setw(15)<<"Gender"<<setw(15)<<"Age"<<"\n"; //format 15 space columns
    }
    ifstream myFile;
    myFile.open("input.txt");
    if (myFile.is_open())
    {
        while( getline(myFile,line))
        {
            cout << line << "\n";
        }
    }
    return 0;
}