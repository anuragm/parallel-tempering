/**
 *   \file pt.cpp
 *   \brief Implementation of pt.cpp
 *
 *
 */

#include "pt.hpp"
#include <iostream>
#include <armadillo>
#include <boost/dynamic_bitset.hpp>
#include <cmath>
#include <memory>

//Redefine global constants.
namespace pt{
    std::mt19937 rand_eng(time_seed);
    std::uniform_real_distribution<double> uniform_dist =
        std::uniform_real_distribution<double>(0.0,1.0);
}

/**
 *  \brief Reads a text file for Ising Hamiltonian
 *
 *  The file must be in format \f$ (i\; j\; J_{ij}) \f$ where \f$ i=j \f$ specifies
 *  local field \f$ h_i \f$.
 *
 *  \param fileName is a C++ string with the name of the file to be read.
 *  \return None
 */
void pt::Hamiltonian::read_file(const std::string& fileName){

    std::ifstream hamFile;
        hamFile.open(fileName.c_str());

    if(!hamFile.is_open()) //If file is not opened, return silently.
    {
        std::cerr<<"cannot read from Hamiltonian file."<<
            " Check if the file exists and is readable \n";
        std::logic_error("Cannot read Hamiltonian file");
        return;
    }

    //Initialise size of local fields and couplings.
    h.set_size(num_of_qubits);
    J.set_size(num_of_qubits,num_of_qubits);

    //For each line, read the file into h and J's.
    while(!hamFile.eof())
    {
        int location1, location2;
        hamFile>>location1>>location2;
        if (location1==location2)
            hamFile>>h(location1);
        else
        {
            //make sure J is initialized as upper triangle matrix
            int rowLocation = (location1<location2)?location1:location2;
            int colLocation = location1+location2-rowLocation;
            hamFile>>J(rowLocation,colLocation);
        }
    }

    //Done!. Care must be taken to make J symmetric.
    J = J + J.t();
    hamFile.close();

} //end of read_file

/**
 *  \brief Returns the energy of the current state of system
 *
 *  Computes energy by performing the QUBO calculation and returns the equivalent Ising
 *  energy.
 *
 *  \return double precision energy
 */

double pt::SimulatedAnnealing::get_energy() const{
    //Remember, state is bitstring, and not a array of 0 and 1.
    double energy = 0;

    energy -= ham.get_offset();

    arma::mat col_temp = (ham.get_Q()*state).t();
    energy += arma::as_scalar(col_temp*state);

    return energy;
}

/**
 *  \brief Perform single step of simulated annealing
 *
 *  Modifies the internal state by doing a single Monte-Carlo step.
 *
 *  \return None
 */
void pt::SimulatedAnnealing::anneal(){

    double old_energy, new_energy, prob_to_flip;
    int qubit_to_flip;

    qubit_to_flip = rand_qubit(pt::rand_eng);
    old_energy = get_energy();
    state.flip(qubit_to_flip);
    new_energy = get_energy();

    prob_to_flip = std::exp(-beta*(new_energy-old_energy));
    if(prob_to_flip > pt::uniform_dist(pt::rand_eng) )
        return;
    else
        state.flip(qubit_to_flip);
}

/**
 *  \brief Converts Ising to QUBO
 *
 *  Since the internal calculation of energy depends on calculation via QUBO matrix and C++
 *  bitstring state, it is required to convert h and J to Q and offset every time h and J are
 *  changed for the object
 *
 *  \return None
 */
void pt::Hamiltonian::computeQUBO(){

    offset = -arma::sum(h) + arma::accu(J);
    Q.set_size(num_of_qubits,num_of_qubits);
    std::printf("Number of rows of Q is %llu and columns is %llu \n",Q.n_rows,Q.n_cols);

    for(arma::uword ii=0;ii<num_of_qubits;ii++)
        for(arma::uword jj=0;jj<num_of_qubits;jj++)
            if(ii==jj)
                Q(ii,ii) = 2*(h(ii) - arma::sum(J.row(ii)) - arma::sum(J.col(ii)));
            else
                Q(ii,jj) = 4*J(ii,jj);

}

/**
 *  \brief Overloaded multiplication operator for multiplying arma::mat for BitString
 *
 *  A overloaded template operator is provided to easily multiply a matrix with BitString. This
 *  is done so that the calculation for energy of a particular state with the given Hamiltonian
 *  appears in a more natural form.
 *
 *  \param in_mat : The matrix to be multiplied
 *  \param in_state: The state on which the Matrix is operated on
 *  \return A column vector of result of the multiplication.
 */
template <class T>
const arma::Mat<T> operator*
(const arma::Mat<T>& in_mat,const pt::BitString& in_state){

    arma::Mat<T> result(in_mat.n_rows,1,arma::fill::zeros);
    arma::uvec non_zero_elements;
    boost::dynamic_bitset<> in_bitset = in_state.get_bitset();

    if(~in_bitset.any())  //If no bits are 1, multiplication is zero.
        return result;

    for(arma::uword ii=0;ii<in_bitset.size();ii++)
        if(in_bitset[ii])
            non_zero_elements << ii;

    result = arma::sum(in_mat.cols(non_zero_elements),1);
    return result;
}

pt::ParallelTempering::ParallelTempering
(const pt::Hamiltonian& in_ham,arma::uword in_num_of_instances):
    num_of_instances(in_num_of_instances),ham(in_ham){
    beta =  arma::vec(num_of_instances,arma::fill::zeros);
    common_init();
}

pt::ParallelTempering::ParallelTempering(const Hamiltonian& in_ham):
    ParallelTempering(in_ham,64){}

pt::ParallelTempering::ParallelTempering(const pt::Hamiltonian& in_ham,
                    double in_base_beta, double in_final_beta):
    ham(in_ham),base_beta(in_base_beta),final_beta(in_final_beta){
    num_of_instances = ham.size();
    beta  = arma::vec(num_of_instances,arma::fill::zeros);
    common_init();
}

pt::ParallelTempering::ParallelTempering
(const Hamiltonian& in_ham, const BitString& in_state):
    ham(in_ham){
    num_of_instances = ham.size();
    beta  = arma::vec(num_of_instances,arma::fill::zeros);
    common_init();
    //Set each state to the input state.
    for (auto &ii: instances)
        ii->set_state(in_state);
}

pt::ParallelTempering
::ParallelTempering
(const Hamiltonian& in_ham, const BitString& in_state,
 const arma::vec& in_temperature):
    ham(in_ham),beta(1.0/in_temperature){
    common_init();
    //Set each state to the input state.
    for (auto &ii: instances)
        ii->set_state(in_state);
}


pt::ParallelTempering::~ParallelTempering(){
    //Release all the unique pointer.
    for(auto &ii: instances)
        ii.reset();
}


void pt::ParallelTempering::common_init(){
    //create instances and proper space
    instances
        = std::vector<std::unique_ptr<SimulatedAnnealing>>(num_of_instances);

    //Assign the same Hamiltonian to all SA instances.
    for (auto& ii: instances)
        ii.reset(new pt::SimulatedAnnealing(ham));

    //If no temperature was allocated, set it in geometric progression.
    if( !arma::any(beta)){

        double beta_ratio = std::pow(final_beta/base_beta,1/(num_of_instances-1));
        beta(0) = base_beta;
        beta(num_of_instances-1) = final_beta;
        for(arma::uword ii=1;ii<(num_of_instances-2);ii++)
            beta(ii) = base_beta*std::pow(beta_ratio,ii);
    }

    //Set this as temperature of each SA object
    for(arma::uword ii=0;ii<num_of_instances;ii++)
        instances[ii]->set_beta(beta(ii));
}
