/**
 *   \file bitstring.hpp
 *   \brief Interface for the various classes used in the parallel
 *   tempering algorithm.
 *
 *   All definitions in this interface belong to the namespace
 *   pt. Different classes provide for different functions. Due to use of templating, the
 *   definitions of the functions are contained in this file as well.
 *
 */

#ifndef BITSTRING_HPP
#define BITSTRING_HPP

#include <cstddef>
#include <armadillo>
#include <bitset>
#include <random>
#include <chrono>
#include <cmath>

namespace pt{

    //Global variables
    const double dw_temperature=1.383;
    const arma::uword dw_numberOfQubits=512;

    const auto time_seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 rand_eng(time_seed);
    std::uniform_real_distribution<double> uniform_dist =
        std::uniform_real_distribution<double>(0.0,1.0);

    //Class definitions
    template <arma::uword numOfQubits=dw_numberOfQubits>
    class BitString{
    private:
        std::bitset<numOfQubits> state;

    public:
        const std::bitset<numOfQubits>& get_bitset() const{return state;}
        arma::uword size() const { return numOfQubits;}
        void flip(unsigned int bit){ state.flip(bit);}

    }; //end class BitString

    template <arma::uword numOfQubits=dw_numberOfQubits>
    class Hamiltonian{
    private:
        arma::mat J = arma::zeros<arma::mat>(numOfQubits,numOfQubits);
        arma::vec h = arma::zeros<arma::vec>(numOfQubits);
        arma::mat Q = arma::zeros<arma::mat>(numOfQubits,numOfQubits);
        double offset;

        void computeQUBO();
    public:
        arma::uword size() const {return numOfQubits;}
        const arma::mat& get_J() const {return J;}
        const arma::vec& get_h() const {return h;}
        double get_offset() const {return offset;}
        const arma::mat& get_Q() const {return Q;}
        void read_file(const std::string&);

        Hamiltonian(){};
        Hamiltonian(arma::vec in_h,arma::mat in_J):
            h(in_h),J(in_J) { computeQUBO();}
        Hamiltonian(const std::string& fileName){
            read_file(fileName);
        }

    }; //end class Hamiltonian

    template <arma::uword numOfQubits=dw_numberOfQubits>
    class SimulatedAnnealing{
    private:

        std::uniform_int_distribution<int> rand_qubit =
            std::uniform_int_distribution<int>(0,numOfQubits-1);
        Hamiltonian<numOfQubits> ham;
        BitString<numOfQubits> state; ///Initialises to zero
        double beta = 1/dw_temperature;

    public:
        SimulatedAnnealing(const Hamiltonian<numOfQubits>& in_ham):
            ham(in_ham) {} ;
        SimulatedAnnealing(const Hamiltonian<numOfQubits>& in_ham,
                           const BitString<numOfQubits>& in_state):
            ham(in_ham), state(in_state) {};
        SimulatedAnnealing(const Hamiltonian<numOfQubits>& in_ham,
                           const BitString<numOfQubits>& in_state,
            double in_beta):
            ham(in_ham), state(in_state), beta(in_beta) {};

        double get_temperature() const {return 1/beta;}
        void set_temperature(double in_temperature){
            beta = 1/in_temperature;
        }

        double get_beta() const {return beta;}
        void set_beta(const double& in_beta){beta=in_beta;}

        BitString<numOfQubits> get_state() const {return state;}
        void set_state(const BitString<numOfQubits>& in_state){state=in_state;}

        double get_energy() const;
        void anneal();
    }; //end class Simulated Annealing

} //end namespace pt.

/// Defining a overloaded operator for multiplying arma::mat to BitString.
template <arma::uword numOfQubits, class T>
const arma::Mat<T> operator* (const arma::Mat<T>&,const pt::BitString<numOfQubits>&);
///////////////////////////////////////////////////////////////////////////////////////////////

/**
 *  \brief Reads a text file for Ising Hamiltonian
 *
 *  The file must be in format \f$ (i\; j\; J_{ij}) \f$ where \f$ i=j \f$ specifies
 *  local field \f$ h_i \f$.
 *
 *  \param fileName is a C++ string with the name of the file to be read.
 *  \return None
 */
template <arma::uword numOfQubits>
void pt::Hamiltonian<numOfQubits>::read_file(const std::string& fileName){

    std::ifstream hamFile;

    if (fileName.empty())
        hamFile.open("hamiltonian.config");
    else
        hamFile.open(fileName.c_str());

    if(!hamFile.is_open()) //If file is not opened, return silently.
    {
        std::cerr<<"cannot read from Hamiltonian file."<<
            " Check if the file exists and is readable \n";
        std::logic_error("Cannot read Hamiltonian file");
        return;
    }

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

template <arma::uword numOfQubits>
double pt::SimulatedAnnealing<numOfQubits>::get_energy() const{
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
template <arma::uword numOfQubits>
void pt::SimulatedAnnealing<numOfQubits>::anneal(){

    double old_energy, new_energy, prob_to_flip;
    int qubit_to_flip;

    qubit_to_flip = rand_qubit(rand_eng);
    old_energy = get_energy();
    state.flip(qubit_to_flip);
    new_energy = get_energy();

    prob_to_flip = std::exp(-beta*(new_energy-old_energy));
    if(prob_to_flip > uniform_dist(rand_eng) )
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
template <arma::uword numOfQubits>
void pt::Hamiltonian<numOfQubits>::computeQUBO(){

    offset = -arma::sum(h) + arma::accu(J);
    for(int ii=0;ii<numOfQubits;ii++)
        for(int jj=0;jj<numOfQubits;jj++)
            if(ii==jj)
                Q(ii,ii) = 2*(h(ii) - arma::sum(J.row(ii)) - arma::sum(J.col(ii)));
            else
                Q(ii,jj) = 4*J(ii,jj);
}

/// Defining a overloaded operator for multiplying arma::mat to BitString.
template <arma::uword numOfQubits, class T>
const arma::Mat<T> operator*
(const arma::Mat<T>& in_mat,const pt::BitString<numOfQubits>& in_state){

    arma::Mat<T> result(in_mat.n_rows,1,arma::fill::zeros);
    arma::uvec non_zero_elements;
    std::bitset<numOfQubits> in_bitset = in_state.get_bitset();

    for(int ii=0;ii<numOfQubits;ii++)
        if(in_bitset[ii])
            non_zero_elements << ii;

    if(non_zero_elements.is_empty()) //If no bits are 1, multiplication is zero.
        return result;

    result = arma::sum(in_mat.cols(non_zero_elements),1);
    return result;
}

#endif //bitstring.hpp
