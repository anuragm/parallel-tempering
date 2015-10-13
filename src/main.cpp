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

    std::cout << "Double epsilon on this machine is "<<
        std::numeric_limits<double>::epsilon()<<std::endl;
    std::cout << "Long Double epsilon on this machine is "<<
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
    std::cout << "Number of PT Swaps is "  <<num_pt_swaps  <<std::endl;
    std::cout << "Number of qubits is "    <<num_of_qubit  <<std::endl;
    std::cout << "File to read from is "   <<file_name     <<std::endl;
    std::cout << "Offset is "              <<offset        <<std::endl;
    std::cout << "Tolerance is "           <<tolerance     <<std::endl;
    pt::Hamiltonian temp_ham(file_name,num_of_qubit,offset);

    pt::ParallelTempering temp_pt(temp_ham,64);
    temp_pt.set_num_of_SA_anneal(num_sa_anneals);
    temp_pt.set_num_of_swaps(num_pt_swaps);
    temp_pt.set_beta_range(pt::DW_BETA,pt::DW_BETA/2);

//    pt::PTTestThermalise pt_thermal_test(num_of_qubit,64,tolerance);
//    temp_pt.push(&pt_thermal_test);

    std::cout << "Now performing parallel tempering for initial thermalisation\n";
    arma::wall_clock timer;
    //First, let the system thermalize.
    double swap_time=0, anneal_time=0;
    unsigned long swap_count=0;
    while(swap_count<3*num_pt_swaps){
        timer.tic(); temp_pt.perform_anneal(); anneal_time += timer.toc();
        timer.tic(); temp_pt.perform_swap();   swap_time += timer.toc();
        swap_count++;
        std::cout << "Completed "<<swap_count<<" swaps.\r";
    }
     std::cout<< "SA anneal time was "<<anneal_time<<" and swap time was "<<swap_time
             <<" seconds \n";
     std::cout <<"beta values used were "<<temp_pt.get_beta().t()<<std::endl;

     //Now do test run to check if system did thermalise.
     std::cout << "Now doing thermalisation testing\n";
     swap_count=0; swap_time=0; anneal_time=0;
     temp_pt.reset_status();
     while(swap_count<num_pt_swaps){
         timer.tic(); temp_pt.perform_anneal(); anneal_time += timer.toc();
         timer.tic(); temp_pt.perform_swap();   swap_time += timer.toc();
         swap_count++;
         std::cout << "Completed "<<swap_count<<" swaps.\r";
     }
     std::cout<< "SA anneal time was "<<anneal_time<<" and swap time was "<<swap_time
              <<" seconds \n";

     //Now print the status.
     temp_pt.status();

     return 0;
}
