/*  l0129.cpp
    header example
*/

#include "l0129.hpp"

int main()
{
    string line;
    cout << "Hello world ! \n";
    fstream myFile("input.txt");
    if(myFile.is_open())
    {
        while( getline(myFile, line))
        {
            cout << "File: " << line << "\n";
        }
    }
    cout << "How old are you?\n";
    string age_s_;
    float age_;
    getline(cin, age_s_);
    stringstream(age_s_) >> age_;
    cout << "You are " << age_ << " years old\n";
    age_ = age_ + 10;
    cout << "In 10 years you'll be " << age_ << "\n";
    
    return 0;
}

/*
wed 18 eoday
need sig
778944843656
*/