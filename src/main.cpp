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
#include <tclap/CmdLine.h>

void get_params(int,char**,std::string&,arma::uword&,arma::uword&,arma::uword&);

int main(int argc, char** argv){

    arma::uword num_sa_anneals, num_pt_swaps,num_of_qubit;
    std::string file_name;

    get_params(argc,argv,file_name,num_sa_anneals,num_pt_swaps,num_of_qubit);
    std::cout << "Number of SA anneals is "<<num_sa_anneals<<std::endl;
    std::cout << "Number of pt swaps is "<<num_pt_swaps<<std::endl;
    std::cout << "Number of qubits is "<<num_of_qubit<<std::endl;
    std::cout << "File to read from is "<<file_name<<std::endl;

    std::printf("Starting Parallel tempering\n");
    pt::Hamiltonian temp_ham(file_name,num_of_qubit);
    std::cout << "Constructed Hamiltonian \n";
    pt::ParallelTempering temp_pt(temp_ham);
    std::printf("Constructed object. Program complete \n");
    return 0;
}

void get_params(int num_of_args,char** args, std::string& file, arma::uword& num_sa_anneals,
                arma::uword& num_pt_swaps,arma::uword& num_of_qubit){
    try{

        TCLAP::CmdLine cmd("Parallel tempering on Chimera graph", ' ', "0.1");
        TCLAP::ValueArg<unsigned int> sa_runs("s","saruns","Number of SA Runs",false,10,"int");
        TCLAP::ValueArg<unsigned int> num_swaps("n","numswaps","Number of swaps",false,100,
                                                "int");
        TCLAP::ValueArg<unsigned int> num_qubit("q","numqubit","Number of qubits",false,512,
                                                "int");
        TCLAP::UnlabeledValueArg<std::string> ham_file("hamfile","File to anneal",true,"",
                                                       "valid file path");

        cmd.add(sa_runs); cmd.add(num_swaps); cmd.add(ham_file);
        cmd.add(num_qubit);
        cmd.parse(num_of_args,args);

        file           = ham_file.getValue();
        num_sa_anneals = sa_runs.getValue();
        num_pt_swaps   = num_swaps.getValue();
        num_of_qubit   = num_qubit.getValue();
    }
    catch (TCLAP::ArgException& e)
    {
        TCLAP::CmdLine cmd("Command description message", ' ', "0.9");
    }
}
