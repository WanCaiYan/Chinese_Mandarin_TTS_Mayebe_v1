#include <cstdio>
#include <iostream>
#include <iostream>
#include <fstream>
#include <string>


using namespace std;

int main() {
    ofstream fout;


    fout.open("t555.txt");
    fout<<"one time"<<endl;
    fout.close();
    return 0;
}
