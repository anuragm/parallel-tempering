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
 *  \param fileName: is a C++ string with the name of the file to be read.
 *  \param offset : is the offset form numbering the first qubit as 0. For example, if the
 *  qubit numbering starts from 1 in input file, offset is -1.
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
} //end of read_file

/**
 *  \brief Returns the energy of a state of system
 *
 *  Computes energy by performing the QUBO calculation and returns the equivalent Ising
 *  energy.
 *
 *  \param bit_state: the state whose energy has to be calculated.
 *  \return double precision energy
 */

double pt::Hamiltonian::get_energy(const boost::dynamic_bitset<>& bit_state) const{
    //Remember, state is bitstring, and not a array of 0 and 1.
    typedef std::string::size_type size_type;
    double energy = 0;
    std::vector<arma::uword> non_zero_indices;
    for(size_type ii=0; ii<bit_state.size();ii++){
        if(bit_state[ii])
            non_zero_indices.push_back(ii);
    }
    arma::uvec non_zero_locations(non_zero_indices);

    energy = offset
        + 0.5*arma::accu(Q.submat(non_zero_locations,non_zero_locations));

    return energy;
}

/**
 *  \brief Calculates the energy difference due to flipping a bit
 *
 *  Given a bitstring and the qubit to be flipped, the energy difference can be calculated by
 *  simply calculating the change in the local neighbourhood. This helps in quickly calculating
 *  the probability of one qubit flip.
 *
 *  \param qubit_number: The qubit to be flipped
 *  \param bitstring   : The bitstring whose qubit has been flipped.
 *  \return the energy difference between final state after flip and initial state before
 *  flip. \f$ \Delta E = E_f - E_i \f$.
 */
double pt::Hamiltonian::flip_energy_diff(int qubit_number,
                                         const boost::dynamic_bitset<>& bitstring){
    //Energy diff is sum all Q(ii,jj) s.t. jj is 1.
    double diff_energy = 0;
    const arma::uvec& current_ngbrs = get_ngbrs(qubit_number);
    if(current_ngbrs.is_empty())
        return 0;

    for(arma::uword jj=0;jj<current_ngbrs.n_elem;jj++){
        diff_energy +=
            (Q(qubit_number,current_ngbrs(jj))*bitstring[current_ngbrs(jj)]);
    }

    //If the bit was flipped from 0 to 1, energy is added. If it flipped from 1 to 0, energy is
    //subtracted. Adjust for the local field Q(ii,ii) thing as well.
    if(bitstring[qubit_number])
        diff_energy = -1.0*(diff_energy-0.5*Q(qubit_number,qubit_number));
    else
        diff_energy = diff_energy + 0.5*Q(qubit_number,qubit_number);

    return diff_energy;
}

/**
 *  \brief Pre-computes the QUBO matrix.
 *
 *  Since the internal calculation of energy depends on calculation via QUBO matrix and C++
 *  bitstring state, it is required to convert h and J to Q and offset every time h and J are
 *  changed for the object
 *
 *  \return None
 */
void pt::Hamiltonian::pre_compute(){

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
 *  \brief Computes the energy difference between two bit-strings
 *
 *  The energy difference between two bitstring can be simply calculated by noting the spins
 *  that they differ at, and summing the change in energy that are caused by flipping those
 *  bits.
 *
 *  \param a : The first bit-string
 *  \param b : The second bit-string
 *  \return the difference in energy \f$ E_a - E_b \f$
 */
double pt::Hamiltonian::energy_diff
(boost::dynamic_bitset<> a, const boost::dynamic_bitset<>& b){
    typedef std::string::size_type size_type;
    auto diff_bitset(a ^ b); //Bitwise XOR
    double energy_diff = 0;
    //Add flip difference as we move from 'a', the final state, to 'b', the initial state.
    for(size_type ii=0;ii<diff_bitset.size();ii++){
        if (diff_bitset[ii]){
            energy_diff -= flip_energy_diff(ii,a);
            a.flip(ii);
        }
    }

    return energy_diff;
}

//Parallel tempering constructors.
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
                                         arma::uword in_num_of_instances):
    num_of_instances(in_num_of_instances),ham(in_ham),base_beta(in_base_beta),
    final_beta(in_final_beta){
    beta  = arma::vec(num_of_instances,arma::fill::zeros);
    common_init();
}

pt::ParallelTempering::ParallelTempering
(const Hamiltonian& in_ham, const arma::vec& in_temperature):
    ham(in_ham),beta(1.0/in_temperature){
    common_init();
}

pt::ParallelTempering::~ParallelTempering(){
    //Release all the unique pointer.
    for(auto &ii: instances)
        ii.reset();
}

void pt::ParallelTempering::common_init(){
    //create instances and proper space
    instances = std::vector<std::unique_ptr<boost::dynamic_bitset<>>>(num_of_instances);
    for (auto& ii: instances)
        ii.reset(new boost::dynamic_bitset<>(ham.size()));

    //Allocate the random number generator
    rand_qubit = std::uniform_int_distribution<int>(0,ham.size()-1);
    //If no temperature was allocated, set it in geometric progression.
    arma::vec log_beta(num_of_instances);
    if( !arma::any(beta)){
        double log_beta_ratio = std::log(final_beta/base_beta)/double(num_of_instances-1);
        for(arma::uword ii=0;ii<(num_of_instances);ii++){
            log_beta(ii) = ii*log_beta_ratio + std::log(base_beta);
        }
    }
    beta = arma::exp(log_beta);
}

/**
 *  \brief Perform Simulated annealing on all the instances
 *
 *  One step of parallel tempering is to perform Monte Carlo anneal on each instance. This is
 *  done by flipping a random qubit, and accepting the flip according to Metropolis rule.
 *
 *  \param anneal_steps is an unsigned integers denoting number of steps each instance should
 *  be annealed. If no value is specified, we perform the default number of annealing steps.
 */

void pt::ParallelTempering::perform_anneal(arma::uword anneal_steps){
    //For all instances, to anneal_steps number of SA steps.
    for(arma::uword ii_anneal=0;ii_anneal<anneal_steps;ii_anneal++){
        for(arma::uword ii_instance=0;ii_instance<num_of_instances;ii_instance++){
            auto& state = instances[ii_instance];
            int qubit_to_flip = rand_qubit(rand_eng);
            double energy_diff = ham.flip_energy_diff(qubit_to_flip,*state);
            double prob_to_flip = std::exp(-beta(ii_instance)*energy_diff);
            if(prob_to_flip > uniform_dist(rand_eng))
                state->flip(qubit_to_flip);
        }
    }
}

/**
 *  \brief Swaps various instances in the PT exchange process.
 *
 *  After we have annealed for few steps each instance, we calculate the probability of
 *  exchanging two instance and then do so via a Metropolis update. Exchanging instances
 *  ensures that the algorithm can explore phase space at multiple temperature, while ensuring
 *  detailed balance.
 *
 */
void pt::ParallelTempering::perform_swap(){
    double beta1, beta2;
    auto rand_num = std::bind(uniform_dist,rand_eng);

    for(arma::uword ii=0; ii<(num_of_instances-1);ii++){
        beta1 = beta(ii);
        beta2 = beta(ii+1);
        double energy_diff = ham.energy_diff(*instances[ii+1],*instances[ii]);
 
        double swap_prob = std::exp(-(beta2-beta1)*energy_diff);
        bool should_swap = (rand_num()<swap_prob);

        if(should_swap){ //Swap instances ii and ii+1 in instances.
            std::swap(instances[ii],instances[ii+1]); //Cheap, because only pointers change.
            beta(ii) = beta2;
            beta(ii+1) = beta1;
        }
    }
}

arma::vec pt::ParallelTempering::get_energies() const{
    arma::vec energies(num_of_instances);
    for(arma::uword ii=0;ii<num_of_instances;ii++)
        energies(ii) = ham.get_energy(*instances[ii]);

    return energies;
}
