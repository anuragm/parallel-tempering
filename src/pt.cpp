/**
 *   \file pt.cpp
 *   \brief Implementation of pt.hpp
 *
 *
 */

#include "pt.hpp"
#include <iostream>
#include <armadillo>
#include <boost/dynamic_bitset.hpp>
#include <cmath>
#include <memory>
#include <climits>
#include <cassert>
#include <gnuplot-iostream.h>

///////////////////////////////////////////////////////////////////////////////////////////////
/**
 *  \brief Creates a random bitset
 *
 *  Each bit set is stored in blocks, which are usually integers of some type. We create a
 *  bunch of random numbers for that type using the generator, and then copy the created
 *  numbers to the input bitset.
 *
 *  \param bitset - The bit set which has to be set to random bit string.
 */

void make_random_bitset(pt::boost_bitset& bitset){
    using sizetype = std::string::size_type;
    auto num_of_bits = bitset.size();
    auto cointoss = std::bind(std::bernoulli_distribution(0.5),pt::rand_eng);
    for(sizetype ii=0;ii<num_of_bits;ii++)
        bitset[ii] = cointoss();
}
///////////////////////////////////////////////////////////////////////////////////////////////
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
///////////////////////////////////////////////////////////////////////////////////////////////
/**
 *  \brief Returns the energy of a state of system
 *
 *  Computes energy by performing the QUBO calculation and returns the equivalent Ising
 *  energy.
 *
 *  \param bit_state: the state whose energy has to be calculated.
 *  \return double precision energy
 */

double pt::Hamiltonian::get_energy(const boost_bitset& bit_state) const{
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
///////////////////////////////////////////////////////////////////////////////////////////////
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
                                         const boost_bitset& bitstring){
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
///////////////////////////////////////////////////////////////////////////////////////////////
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

    //Scale down Hamiltonian if required.
    scale_to_unit();

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
///////////////////////////////////////////////////////////////////////////////////////////////
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
(boost_bitset a, const boost_bitset& b){
    typedef std::string::size_type size_type;
    pt::boost_bitset diff_bitset = (a ^ b); //Bitwise XOR
    double energy_diff = 0;
    //Add flip difference as we move from 'a', the final state, to 'b', the initial state.
    for(size_type ii=0;ii<diff_bitset.size();ii++){
        if (diff_bitset[ii]){
            energy_diff -= flip_energy_diff(ii,a);
            a.flip(ii);
        }
    }
    //a.reset(); //Workaround needed for boost to not complain.
    return energy_diff;
}
///////////////////////////////////////////////////////////////////////////////////////////////

void pt::Hamiltonian::scale_to_unit(){
    double max_h = arma::abs(h).max();
    double max_J = arma::abs(J).max();
    double scaleFactor = ((max_h>max_J)?max_h:max_J);

    if(scaleFactor>1.0){
        h = h/scaleFactor;
        J = J/scaleFactor;
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////
//Parallel tempering constructors.

pt::ParallelTempering::ParallelTempering
(const pt::Hamiltonian& in_ham,arma::uword in_num_of_instances):
    num_of_instances(in_num_of_instances),ham(in_ham),
    current_tags(num_of_instances){

    //Set up various default values.
    beta               = arma::vec(num_of_instances,arma::fill::zeros);
    base_beta          = DW_BETA;
    final_beta         = DW_BETA/5;
    num_of_SA_anneal   = 10;
    num_of_swaps       = 100;
    anneal_counter     = 0;
    swap_counter       = 0;
    flag_init          = false;
    tag_frequency.set_size(num_of_instances,num_of_instances);
    tag_frequency.zeros();

    //Use accuracy of 0.01 for binning overlap probabilities.
    prob_overlap.reserve(num_of_instances);
    for(arma::uword ii=0;ii<num_of_instances;ii++)
        prob_overlap.push_back(OverlapHistogram(200));

    //Memory to be initialised just before first anneal is performed. This is done by calling
    //init function.
}

///////////////////////////////////////////////////////////////////////////////////////////////

pt::ParallelTempering::~ParallelTempering(){
    //Release all the unique pointers.
    for(auto &ii: instances1){
        ii.reset();
    }

    for(auto &ii: instances2){
        ii.reset();
    }

}
///////////////////////////////////////////////////////////////////////////////////////////////

void pt::ParallelTempering::init(){

    //create instances and proper space, and then initialise them to random bit strings.
    instances1 = std::vector<std::unique_ptr<boost_bitset>>(num_of_instances);
    instances2 = std::vector<std::unique_ptr<boost_bitset>>(num_of_instances);

    for (auto& ii: instances1){
        ii  = std::make_unique<boost_bitset>(ham.size());
        make_random_bitset(*ii);
    }
    for (auto& ii: instances2){
        ii  = std::make_unique<boost_bitset>(ham.size());
        make_random_bitset(*ii);
    }

    //Initialise energies for these states.
    energies1 = get_energies(INSTANCES_1);
    energies2 = get_energies(INSTANCES_2);

    //Initialise the random qubit distribution.
    rand_qubit = std::uniform_int_distribution<int>(0,ham.size()-1);

    //If no temperature was allocated, set it in geometric progression.
    arma::vec log_beta(beta.n_elem);
    if( !arma::any(beta)){
        double logbeta_ratio = std::log(final_beta/base_beta)/(num_of_instances-1);
        for(arma::uword ii=0;ii<(num_of_instances);ii++){
            log_beta(ii) = ii*logbeta_ratio + std::log(base_beta);
        }
    }
    beta = arma::exp(log_beta);
    
    //Compute the helper quantities for initial vectors.
    for (auto& ii: helper_objects)
        ii->compute(instances1,instances2,energies1,energies2);

    //Set the initial tag array. Each tag is swapped as instances are swapped.
    for(unsigned long ii=0;ii<current_tags.size();ii++){
        current_tags[ii] = ii;
        tag_frequency(ii,ii) = 1; //Before 1st run, instance 'ii' is set to value beta(ii)
    }

    //And now set the init flag to true.
    flag_init = true;
}

///////////////////////////////////////////////////////////////////////////////////////////////

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
    if(!flag_init){
        init();
    }
    arma::uword num_of_qubits = ham.size();

    //Lambda to perform anneal on a state.
    auto anneal_state = [&] (std::unique_ptr<boost_bitset>& state, double& state_energy,
                                             arma::uword ii){
            int qubit_to_flip   = rand_qubit(rand_eng);
            double energy_diff  = ham.flip_energy_diff(qubit_to_flip,*state);
            double prob_to_flip = std::exp(-beta(ii)*energy_diff);
            if (prob_to_flip > uniform_dist(rand_eng)){
                state->flip(qubit_to_flip);
                state_energy += energy_diff;
            }
        };

    //For all instances, to anneal_steps number of SA steps.
    for(arma::uword ii_anneal=0;ii_anneal<anneal_steps;ii_anneal++){
        for(arma::uword ii_instance=0;ii_instance<num_of_instances;ii_instance++){
            //Anneal both copies.
            anneal_state(instances1[ii_instance],energies1(ii_instance),ii_instance);
            anneal_state(instances2[ii_instance],energies2(ii_instance),ii_instance);

            //Compute overlap for each beta and save it.
            boost_bitset overlap_bitset =
                ~((*instances1[ii_instance])^(*instances2[ii_instance]));
            auto block_vector = boost_bitset_to_vector(overlap_bitset);
            double overlap=0;
            for (auto& ii:block_vector)
                overlap += num_of_1bits_fast(ii);
            overlap = 2.0*overlap/num_of_qubits - 1;
            prob_overlap[ii_instance].push_value(overlap);
        }

        anneal_counter++;

        //Compute the helper quantities for annealed vectors.
        for (auto& ii: helper_objects)
            ii->compute(instances1,instances2,energies1,energies2);
    }
}
///////////////////////////////////////////////////////////////////////////////////////////////

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

    auto rand_num = [](){return uniform_dist(rand_eng);};
    auto swapStates = [&] (std::vector<std::unique_ptr<boost_bitset>>& ins,
                           arma::vec& energies, arma::uword ins_number){
        double beta_diff = beta(ins_number+1) - beta(ins_number);
        double energy_diff = ham.energy_diff(*ins[ins_number+1],*ins[ins_number]);
        double log_swap_prob = (beta_diff*energy_diff);
        bool should_swap;
        if(log_swap_prob>0)
            should_swap = true;
        else if(log_swap_prob<0)
            should_swap = (std::log(rand_num())<log_swap_prob);
        else
            should_swap = false;
        if(should_swap){ //Swap instances ii and ii+1 in instances.
            std::swap(ins[ins_number+1],ins[ins_number]); //Cheap, because only pointers change.
            std::swap(energies(ins_number+1),energies(ins_number));
        }
        return should_swap;
    };

    arma::uword swap_accepted=0;
    for(arma::uword ii=0; ii<(num_of_instances-1);ii++){
        bool did_swap;
        did_swap = swapStates(instances1,energies1,ii);
        swapStates(instances2,energies2,ii);
        if(did_swap)
            std::swap(current_tags[ii],current_tags[ii+1]);
        swap_accepted+=did_swap;
    }
    double current_acceptance_ratio = double(swap_accepted)/(num_of_instances-1);
    if(swap_counter==0)
        average_acceptance_ratio = current_acceptance_ratio;
    else
        average_acceptance_ratio =
            (swap_counter*average_acceptance_ratio+current_acceptance_ratio)/
            (swap_counter+1);
    //Now, increment the tag frequencies.
    for(arma::uword ii=0;ii<num_of_instances;ii++)
        tag_frequency(ii,current_tags[ii])++;

    swap_counter++;
}
///////////////////////////////////////////////////////////////////////////////////////////////
/**
 *  \brief Get the energies of current instances
 *
 *  \return A vector of energies.
 */

arma::vec pt::ParallelTempering::get_energies(pt::instance_number ii) const{
    arma::vec energies(num_of_instances);
    if(ii==INSTANCES_1)
        for(arma::uword ii=0;ii<num_of_instances;ii++)
            energies(ii) = ham.get_energy(*instances1[ii]);
    else
        for(arma::uword ii=0;ii<num_of_instances;ii++)
            energies(ii) = ham.get_energy(*instances2[ii]);

    return energies;
}

void pt::ParallelTempering::status() const{
    //First, we should print out the acceptance ratio.
    std::cout << "During "<<swap_counter<<" swaps, the average swap acceptance ratio was "
              <<average_acceptance_ratio<<std::endl;

    //Then, we should report if each P(q) is symmetric.
    for(arma::uword ii=0;ii<num_of_instances;ii++){
        std::cout << "Is P(q) for beta("<<ii<<") symmetric : ";
        std::cout << std::boolalpha<< prob_overlap[ii].is_symmetric(1e-5) <<std::endl;
    }

    //Plot for P(q) for beta(0)
    std::cout << "Here is a histogram for lowest temperature of P(q)\n";
    Gnuplot gp;
    arma::mat hist = prob_overlap[0].get_histogram();
    gp<<"set xrange [-1:1]\n";
    gp<<"plot "<<gp.binFile1d(hist,"record")<<" title 'P(q) for beta(0)' with boxes\n";
//    std::cout << hist <<"\n";

    //And then, let us tell about the how many time each replica visited each temperature.
    std::cout << "Replica (row) visited beta (column) these many times.\n";
    std::cout << "Saved in file tagfreq.txt \n";
    arma::umat tagfreq = arma::trans(tag_frequency);
    tagfreq.save("tagfreq.txt",arma::raw_ascii);
}

void pt::ParallelTempering::reset_status(){
    //Resets the counters for overlap, replica visit and acceptance ratio.
    average_acceptance_ratio = 0;
    swap_counter = 0; //This resets acceptance ratio calculation automatically.

    //Reset overlaps.
    prob_overlap.clear();
    for(arma::uword ii=0;ii<num_of_instances;ii++)
        prob_overlap.emplace_back(200);

    //Reset replica visit array.
    tag_frequency.zeros();
    for(unsigned long ii=0;ii<current_tags.size();ii++){
        current_tags[ii] = ii;
        tag_frequency(ii,ii) = 1; //Before 1st run, instance 'ii' is set to value beta(ii)
    }
}
