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
#include "ptparse.hpp"

int main(int argc, char** argv){

    arma::uword num_sa_anneals, num_pt_swaps,num_of_qubit; int offset;
    std::string file_name;

    cmdParams cmd_params;
    get_cmd_params(&cmd_params,argc,argv);
    num_sa_anneals = cmd_params.num_of_sa_runs;
    num_pt_swaps   = cmd_params.num_of_swaps;
    offset         = cmd_params.qubit_offset;
    file_name      = cmd_params.fileName;
    num_of_qubit   = cmd_params.num_of_qubits;

    std::cout << "Number of SA anneals is "<<num_sa_anneals<<std::endl;
    std::cout << "Number of pt swaps is "  <<num_pt_swaps  <<std::endl;
    std::cout << "Number of qubits is "    <<num_of_qubit  <<std::endl;
    std::cout << "File to read from is "   <<file_name     <<std::endl;
    std::cout << "Offset is "<<offset<<std::endl;

    pt::Hamiltonian temp_ham(file_name,num_of_qubit,offset);
    //pt::SimulatedAnnealing temp_SA(temp_ham);
    //temp_SA.anneal();
    pt::ParallelTempering temp_pt(temp_ham,pt::DW_BETA,pt::DW_BETA/10,64);

    temp_pt.set_num_of_SA_anneal(num_sa_anneals);
    temp_pt.set_num_of_swaps(num_pt_swaps);

    std::cout << "Now performing parallel tempering\n";
    arma::wall_clock timer;

    double swap_time=0, anneal_time=0;
    for(arma::uword ii=0;ii<num_pt_swaps;ii++){
        timer.tic(); temp_pt.perform_anneal(); anneal_time += timer.toc();
        timer.tic(); temp_pt.perform_swap();   swap_time += timer.toc();
    }

    std::cout << "After PT, energy of base beta instances is "
              <<temp_pt.get_energies().at(0)<<std::endl;
    std::cout << "And the minimum energy found at any beta is "
              <<arma::min(temp_pt.get_energies())<<std::endl;
    std::cout<< "SA anneal time was "<<anneal_time<<" and swap time was "<<swap_time
             <<" seconds \n";
    return 0;
}
