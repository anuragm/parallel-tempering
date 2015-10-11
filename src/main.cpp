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
#include "pthelper.hpp"
#include "ptparse.hpp"

int main(int argc, char** argv){

    std::cout << "Double precision on this machine is "<<
        std::numeric_limits<double>::epsilon()<<std::endl;
    std::cout << "Long Double precision on this machine is "<<
        std::numeric_limits<long double>::epsilon()<<std::endl;
    std::cout << "Log of Max value of double precision is "
              <<std::log(std::numeric_limits<double>::max())<<std::endl;
    arma::uword num_sa_anneals, num_pt_swaps,num_of_qubit; int offset;
    std::string file_name;

    cmdParams cmd_params;
    get_cmd_params(&cmd_params,argc,argv);
    num_sa_anneals = cmd_params.num_of_sa_runs;
    num_pt_swaps   = cmd_params.num_of_swaps;
    offset         = cmd_params.qubit_offset;
    file_name      = cmd_params.fileName;
    num_of_qubit   = cmd_params.num_of_qubits;

    double tolerance = 1.0e-8;
    std::cout << "Number of SA anneals is "<<num_sa_anneals<<std::endl;
    std::cout << "Number of qubits is "    <<num_of_qubit  <<std::endl;
    std::cout << "File to read from is "   <<file_name     <<std::endl;
    std::cout << "Offset is "              <<offset        <<std::endl;
    std::cout << "Tolerance is "           <<tolerance     <<std::endl;

    pt::Hamiltonian temp_ham(file_name,num_of_qubit,offset);

    pt::ParallelTempering temp_pt(temp_ham);
    temp_pt.set_num_of_SA_anneal(num_sa_anneals);
    temp_pt.set_num_of_swaps(num_pt_swaps);

    pt::PTTestThermalise pt_thermal_test(num_of_qubit,64,tolerance);
    temp_pt.push(&pt_thermal_test);

    std::cout << "Now performing parallel tempering\n";
    arma::wall_clock timer;

    //First, let the system thermalize.
    double swap_time=0, anneal_time=0;
    unsigned long swap_count=0;
    while(!pt_thermal_test.has_thermalised()){
        timer.tic(); temp_pt.perform_anneal(); anneal_time += timer.toc();
        timer.tic(); temp_pt.perform_swap();   swap_time += timer.toc();
        swap_count++;
        std::cout << "Completed "<<swap_count<<" swaps.\r";
    }
    std::cout<<"Simulation thermalized after "<<swap_count
             <<" Monte Carlo swaps\n";
     std::cout<< "SA anneal time was "<<anneal_time<<" and swap time was "<<swap_time
             <<" seconds \n";

     //Then let it run for 1000 swaps to get energies, and then calculate the autocorrelation
     //length of the energies.
     temp_pt.pop(); //remove the thermal check object.

     //Here push the appropriate object.
     timer.tic();
     pt::PTAutocorrelation pt_corr_obj(100*num_sa_anneals,64);
     temp_pt.push(&pt_corr_obj);
     for(arma::uword ii_swap=0;ii_swap<100;ii_swap++){
         temp_pt.perform_anneal();
         temp_pt.perform_swap();
     }
     //Get autocorrelation length.
     arma::uword corr_length = pt_corr_obj.get_correlation_length();
     std::cout << "Correlation length was found to be "<<corr_length
               <<" Monte carlo anneals "<<"in "<<timer.toc()<<" seconds.\n";
     temp_pt.pop();

     //And then run and gather data for required number of anneals. Write this separately.
     std::cout << "Now gathering real data\n";
     arma::uword extra_swaps = 1e5/num_sa_anneals;
     pt::PTStore temp_pt_store(corr_length,corr_length*extra_swaps*num_sa_anneals,
                               64,num_of_qubit);
     pt::PTSpinOverlap pt_spin_overlap(64,num_sa_anneals,corr_length*extra_swaps,num_of_qubit);
     temp_pt.push(&temp_pt_store); temp_pt.push(&pt_spin_overlap);
     swap_count=0;
     for(arma::uword ii_swap=0;ii_swap<extra_swaps*corr_length;ii_swap++){
         temp_pt.perform_anneal();
         temp_pt.perform_swap();
         swap_count++;
         std::cout << "Completed "<<swap_count<<"/"<<extra_swaps*corr_length<<" swaps.\r";
     }
     arma::mat energies = temp_pt_store.get_energies();
     std::cout << "\nSize of energy array is ("<<energies.n_rows<<","<<energies.n_cols<<")"
               <<std::endl;
     energies.save("energies.test.txt",arma::raw_ascii);
     std::cout << "Minimum energy found was \n"
               <<arma::conv_to<arma::Col<int>>::from(arma::min(energies,1))
               << std::endl;

     //Plot the graph.
     pt_spin_overlap.plot_to_file_overlap_mean(" ");

     return 0;
}
