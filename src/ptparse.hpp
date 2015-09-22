/**
 *   \file ptparse.hpp
 *   \brief Parser to parse inputs to command line
 *
 *  From the command line, we want to read various parameters on which parallel tempering would
 *  be performed. I will use the tclap header library to parse the inputs. 
 *
 */
#include <tclap/CmdLine.h>

struct cmdParams{
    int num_of_sa_runs;
    int num_of_swaps;
    int num_of_qubits;
    int qubit_offset;
    std::string fileName;
};

void get_cmd_params(cmdParams* cmd_params,int num_of_args,char** args){
        try{

        TCLAP::CmdLine cmd("Parallel tempering on Chimera graph", ' ', "0.1");
        TCLAP::ValueArg<unsigned int> sa_runs("s","saruns","Number of SA Runs",false,10,"int");
        TCLAP::ValueArg<unsigned int> num_swaps("n","numswaps","Number of swaps",false,100,
                                                "int");
        TCLAP::ValueArg<unsigned int> num_qubit("q","numqubit","Number of qubits",false,512,
                                                "int");
        TCLAP::ValueArg<int> q_offset("o","offset","Qubit numbering offset",false,0,"int");
        TCLAP::UnlabeledValueArg<std::string> ham_file("hamfile","File to anneal",true,"",
                                                       "valid file path");

        cmd.add(sa_runs); cmd.add(num_swaps); cmd.add(ham_file);
        cmd.add(num_qubit); cmd.add(q_offset);
        cmd.parse(num_of_args,args);

        cmd_params->fileName       = ham_file.getValue();
        cmd_params->num_of_sa_runs = sa_runs.getValue();
        cmd_params->num_of_swaps   = num_swaps.getValue();
        cmd_params->num_of_qubits  = num_qubit.getValue();
        cmd_params->qubit_offset   = q_offset.getValue();
    }
    catch (TCLAP::ArgException& e)
    {
        TCLAP::CmdLine cmd("Command description message", ' ', "0.9");
    }
}
