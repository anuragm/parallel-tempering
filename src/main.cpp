/**
 *   \file  main.cpp
 *   \brief Runs the main algorithm for parallel tempering
 *
 *  In this main file, we want to apply the various functions defined
 *  elsewhere to do parallel tempering.
 *
 */

#include <iostream>
#include <armadillo>
#include <climits>
#include "pt.hpp"

int main(){

    pt::Hamiltonian<8> test_ham;
    test_ham.read_file("hamiltonian.txt");
    std::cout << "Successfully read file \n";

    std::cout << "Initialising a parallel tempering object of 24 instances and 8 qubits \n";
    pt::ParallelTempering<24,8> test_PT(test_ham);
    std::cout << "Object successfully initialised.\n";

    return 0;
}
