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
void pt::Hamiltonian::read_file(const std::string& fileName, int offset){

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
    h = arma::vec(num_of_qubits,arma::fill::zeros);
    J = arma::mat(num_of_qubits,num_of_qubits,arma::fill::zeros);

    //For each line, read the file into h and J's.
    while(!hamFile.eof())
    {
        int location1, location2;
        hamFile>>location1>>location2;
        if (location1==location2)
            hamFile>>h(location1+offset);
        else
        {
            //make sure J is initialized as upper triangle matrix
            int rowLocation = (location1<location2)?location1:location2;
            int colLocation = location1+location2-rowLocation;
            hamFile>>J(rowLocation+offset,colLocation+offset);
        }
    }

    //Done!. Care must be taken to make J symmetric.
    J = J + J.t();
    hamFile.close();
    computeQUBO();

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
    typedef std::string::size_type size_type;
    double energy = 0;
    std::vector<arma::uword> non_zero_indices;
    boost::dynamic_bitset<> bit_state = state.get_bitset();
    for(size_type ii=0; ii<bit_state.size();ii++){
        if(bit_state[ii])
            non_zero_indices.push_back(ii);
    }
    arma::uvec non_zero_locations(non_zero_indices);

    energy = ham.get_offset()
        + 0.5*arma::accu(ham.get_Q().submat(non_zero_locations,non_zero_locations));

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
    int qubit_to_flip = rand_qubit(rand_eng);
    double energy_diff = flip_energy_diff(qubit_to_flip);
    double prob_to_flip = std::exp(-beta*energy_diff);
    if(prob_to_flip > uniform_dist(rand_eng))
        state.flip(qubit_to_flip);
}

double pt::SimulatedAnnealing::flip_energy_diff(int qubit_number){
    //Energy diff is sum all Q(ii,jj) s.t. jj is 1.
    double diff_energy = 0;
    const boost::dynamic_bitset<>& current_bitstring = state.get_bitset();
    const arma::uvec& current_ngbrs = ham.get_ngbrs(qubit_number);
    if(current_ngbrs.is_empty())
        return 0;

    const arma::mat& Q = ham.get_Q();
    for(arma::uword jj=0;jj<current_ngbrs.n_elem;jj++){
        diff_energy +=
            (Q(qubit_number,current_ngbrs(jj))*current_bitstring[current_ngbrs(jj)]);
    }

    //If the bit was flipped from 0 to 1, energy is added. If it flipped from 1 to 0, energy is
    //subtracted. Adjust for the local field Q(ii,ii) thing as well.
    if(current_bitstring[qubit_number])
        diff_energy = -1.0*(diff_energy-0.5*Q(qubit_number,qubit_number));
    else
        diff_energy = diff_energy + 0.5*Q(qubit_number,qubit_number);

    return diff_energy;
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

    offset = arma::sum(h) + 0.5*arma::accu(J);
    Q.set_size(num_of_qubits,num_of_qubits);

    Q = 4.0*J;
    for(arma::uword ii=0;ii<num_of_qubits;ii++)
                Q(ii,ii) = 4.0*( -h(ii) - arma::sum(J.col(ii)) );

    //Initialise neighbours.
    neighbours.resize(num_of_qubits);
    for(arma::uword ii=0;ii<num_of_qubits;ii++){
        neighbours[ii] = arma::find(Q.col(ii));
    }
}

/**
 *  \brief Overloaded multiplication operator to perform \f$ Q\,x \f$
 *
 *  A overloaded template operator is provided to easily multiply a matrix with BitString. This
 *  is done so that the calculation for energy of a particular state with the given Hamiltonian
 *  appears in a more natural form.
 *
 *  \param in_mat : The matrix to be multiplied \f$ p \times N \f$
 *  \param in_state: The state on which the Matrix is operated on. \f$ N \times 1 \f$.
 *  \return A column vector of result of the multiplication. \f$ p \times 1 \f$
 */
template <class T>
const arma::Mat<T> operator*
(const arma::Mat<T>& in_mat,const pt::BitString& in_state){

    arma::Mat<T> result(in_mat.n_rows,1,arma::fill::zeros);
    std::vector<arma::uword> non_zero_elements;
    boost::dynamic_bitset<> in_bitset = in_state.get_bitset();

    if(!in_bitset.any())
        return result;  //If no bits are 1, multiplication is zero.

    for(arma::uword ii=0;ii<in_bitset.size();ii++)
        if(in_bitset[ii])
            non_zero_elements.push_back(ii);

    result = arma::sum(in_mat.cols(arma::uvec(non_zero_elements)),1);
    return result;
}

/**
 *  \brief Overloaded multiplication to perform \f$ x^T \,Q \f$
 *
 *  Performs the left matrix multiplication of a matrix with BitString.
 *
 *  \param in_state, which is a bitstring of length N (thought as column vector \f$ Nx1 \f$).
 *  \param in_mat, which is a matrix of \f$ N\times p \f$.
 *  \return A matrix of size \f$ 1\times p\f$, that is, a row vector.
 */
template <class T>
const arma::Mat<T> operator* (const pt::BitString& in_state,const arma::Mat<T>& in_mat){

    arma::Mat<T> result(1,in_mat.n_cols,arma::fill::zeros);
    std::vector<arma::uword> non_zero_elements;
    boost::dynamic_bitset<> in_bitset = in_state.get_bitset();

    if(!in_bitset.any()) //If Bitstring is all zeros
        return result;

    for(arma::uword ii=0;ii<in_bitset.size();ii++)
        if(in_bitset[ii])
            non_zero_elements.push_back(ii);

    result = arma::sum(in_mat.rows(arma::uvec(non_zero_elements)),0);
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
                                         double in_base_beta, double in_final_beta,
                                         int in_num_of_instances):
    num_of_instances(in_num_of_instances),ham(in_ham),base_beta(in_base_beta),
    final_beta(in_final_beta){
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

pt::ParallelTempering::ParallelTempering
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
    //arma::vec log_beta(num_of_instances);
    //if( !arma::any(beta)){
    //    double log_beta_ratio = std::log(final_beta/base_beta)/double(num_of_instances-1);
    //    for(arma::uword ii=0;ii<(num_of_instances);ii++){
    //        log_beta(ii) = ii*log_beta_ratio + std::log(base_beta);
    //    }
    //}
    //beta = arma::exp(log_beta);

    //If no temperature series was allocated, set in linear progression.
    if(!arma::any(beta)){
        for(arma::uword ii=0;ii<num_of_instances;ii++)
            beta(ii) = ii*(final_beta-base_beta)/(num_of_instances-1) + base_beta;
    }

    //Set this as temperature of each SA object
    for(arma::uword ii=0;ii<num_of_instances;ii++)
        instances[ii]->set_beta(beta(ii));
}

void pt::ParallelTempering::perform_anneal(){
    for(auto& ii: instances){
        for(arma::uword jj=0;jj<num_of_SA_anneal;jj++)
            ii->anneal();
    }
}

void pt::ParallelTempering::perform_swap(){
    double beta1, beta2;
    auto rand_num = std::bind(uniform_dist,rand_eng);

    //Calculate the different in temperature and energy.
    arma::vec diff_beta     = arma::diff(beta);
    arma::vec diff_energies = arma::diff(get_energies());

    //Compute exchange probabilities. % is element wise multiplication.
    arma::vec exchange_prob = arma::exp(-diff_beta % diff_energies);
    arma::vec temp_random_num(exchange_prob.n_elem);
    std::generate_n(temp_random_num.begin(), exchange_prob.n_elem, rand_num);
    arma::uvec should_swap  = (exchange_prob>temp_random_num);

    for(arma::uword ii=0; ii<(num_of_instances-1);ii++){
        beta1 = beta(ii);
        beta2 = beta(ii+1);
        if(should_swap(ii)){ //Swap instances ii and ii+1 in instances.
            std::swap(instances[ii],instances[ii+1]); //Cheap, because only pointers change.
            instances[ii]->set_beta(beta2);
            instances[ii+1]->set_beta(beta1);
        }
    }
}

arma::vec pt::ParallelTempering::get_energies() const{
    arma::vec energies(num_of_instances);
    for(arma::uword ii=0;ii<num_of_instances;ii++)
        energies(ii) = instances[ii]->get_energy();

    return energies;
}
