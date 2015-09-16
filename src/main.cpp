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
#include "bitstring.hpp"

int main(){

    const int num_anneal = 5000;
    pt::Hamiltonian<8> test_ham;
    test_ham.read_file("hamiltonian.txt");
    std::cout << "Successfully read file \n";

    pt::SimulatedAnnealing<8> test_SA(test_ham);
    std::cout << "Initial energy for the state is "<<test_SA.get_energy()<<std::endl;
    std::cout << "And the initial state is "<<test_SA.get_state().get_bitset()<<std::endl;
    for(int jj=0;jj<num_anneal;jj++){
        test_SA.anneal();
    }

    std::printf("After %d anneals, the energy is %f. \n",num_anneal,test_SA.get_energy());
    std::cout << "And the state is "<<test_SA.get_state().get_bitset()<<std::endl;
    return 0;
}
