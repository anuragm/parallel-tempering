/**
 *   \file test.cpp
 *   \brief A file to run various test on different functions
 *
 *  We use various Hamiltonian's to asses whether the functions are performing correctly or
 *  not.
 *
 */

#include <iostream>
#include <armadillo>
#include <boost/dynamic_bitset.hpp>
#include <cmath>
#include "../src/pt.hpp"

bool is_equal_double(double a,double b){
    return (std::abs(a-b)<1e-8);
}

int main()
{
    bool energies_test = true;
    bool diff_energies_test = true;

    //Three qubit Hamiltonian test.
    arma::vec three_h(3); arma::mat three_J(3,3);
    three_h(0) = 1; three_h(1) = 1; three_h(2) = 1;
    three_J(0,1) = three_J(1,2) = 1;
    three_J += three_J.t();

    pt::Hamiltonian three_ham(three_h,three_J);
    arma::vec energies = {5,1,-1,-1,1,-3,-1,-1}; //Computed by hand

    //testing energy function.
    for(arma::uword ii=0;ii<8;ii++){
        boost::dynamic_bitset<> bs(3,ii);
        double energy = three_ham.get_energy(bs);
        if(!is_equal_double(energy,energies(ii))){
            energies_test = false;
            std::cerr << "Mismatch of energy for bitstring "<<ii<<"\n";
        }
    }

    //testing difference of energy function between two states.
    for(arma::uword ii=0;ii<8;ii++)
        for(arma::uword jj=0;jj<8;jj++){
            boost::dynamic_bitset<> initial(3,ii);
            boost::dynamic_bitset<> final(3,jj);
            double energy_diff = three_ham.energy_diff(final,initial);
            if(!is_equal_double(energy_diff,energies(jj)-energies(ii))){
                diff_energies_test = false;
                std::cerr<<"Mismatch of energy difference between state "<<ii<<" and "
                         <<jj<<std::endl;
                std::cout << "correct energy is "<<energies(jj)-energies(ii)
                          <<" and function calculated "<<energy_diff<<std::endl;
            }
        }

    //testing if energy difference obtained from flipping a qubit is ok.
    for(unsigned ii=0;ii<8;ii++)
        for(unsigned jj=0;jj<3;jj++)
        {
            unsigned bit_to_flip = jj;
            unsigned flipped_state = ii^(1<<bit_to_flip);
            boost::dynamic_bitset<> initial_state(3,ii);
            boost::dynamic_bitset<> final_state(3,flipped_state);

            double real_energy_diff = energies(flipped_state)-energies(ii);
            double test_energy = three_ham.flip_energy_diff(jj,initial_state);

            if(!is_equal_double(real_energy_diff,test_energy)){
                std::cout << "Energy difference from flipping "<<bit_to_flip
                          <<" qubit of state "<<ii<<" is "<<real_energy_diff
                          <<" while the function gave "<<test_energy<<std::endl;
            }

        }

    if(energies_test && diff_energies_test){
        std::cout << "All test passed successfully \n";
        return 0;
    }
    else{
        std::cerr << "Some test failed.\n";
        return 1;
    }
}
