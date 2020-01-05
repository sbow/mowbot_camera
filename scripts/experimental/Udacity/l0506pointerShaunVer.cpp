/*For this program print for each variable
**print the value of the variable, 
**then print the address where it is stored. 
*/
#include<iostream>
#include<string>
#include<sstream>

int main()
{
    int givenInt;
    float givenFloat;
    double givenDouble ;
    char givenChar;
    std::string givenString;
    std::string derefString;
    
    std::cin >> givenInt;
    std::cin >> givenFloat;
    std::cin >> givenDouble;
    std::cin >> givenChar;
    std::getline(std::cin, givenString);
    
    int * givenInt_p = &givenInt;
    std::cout << "int Value: \t" << * givenInt_p << "\n";
    std::cout << "int Location: \t" << givenInt_p << "\n";
    float * givenFloat_p = &givenFloat;
    std::cout << "flt Value: \t" << * givenFloat_p << "\n";
    std::cout << "flt Location: \t" << givenFloat_p << "\n";    
    double * givenDouble_p = &givenDouble;
    std::cout << "dbl Value: \t" << * givenDouble_p << "\n";
    std::cout << "dbl Location: \t" << givenDouble_p << "\n";    
    char * givenChar_p = &givenChar;
    std::cout << "chr Value: \t" << * givenChar_p << "\n";
    std::cout << "chr Location: \t" << (void*) givenChar_p << "\n"; //note: Overriding pointer type here from (char*) to (void*) to get arround operaor<< overloading behavrious for (char*) arguments to be treated as c_str
    std::string * givenString_p = &givenString;   
    std::cout << "str Value: \t" << givenString << "\n";
    std::cout << "str Location: \t" << givenString_p << "\n";    
    
    return 0;
}
