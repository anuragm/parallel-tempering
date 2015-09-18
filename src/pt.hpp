/**
 *   \file bitstring.hpp
 *   \brief Interface for the various classes used in the parallel
 *   tempering algorithm.
 *
 *   All definitions in this interface belong to the namespace
 *   pt. Different classes provide for different functions.
 *
 */

#ifndef BITSTRING_HPP
#define BITSTRING_HPP

#include <armadillo>
#include <random>
#include <chrono>
#include <cmath>
#include <memory>
#include <boost/dynamic_bitset.hpp>

namespace pt{

    //Global variables
    const double DW_TEMPERATURE=0.10991;
    const double DW_BETA = 1/DW_TEMPERATURE;
    const arma::uword dw_numberOfQubits=512;

    const auto time_seed = std::chrono::system_clock::now().time_since_epoch().count();
    extern std::mt19937 rand_eng;
    extern std::uniform_real_distribution<double> uniform_dist;

    //Class definitions
    class BitString{
    private:
        arma::uword num_of_qubits;
        boost::dynamic_bitset<> state;

    public:
        const boost::dynamic_bitset<>& get_bitset() const{return state;}
        arma::uword size() const { return num_of_qubits;}
        void flip(unsigned int bit){ state.flip(bit);}

        BitString(arma::uword nQubits):
                       num_of_qubits(nQubits){
            state = boost::dynamic_bitset<>(num_of_qubits);
        };
        BitString():BitString(dw_numberOfQubits){};

    }; //end class BitString

    class Hamiltonian{
    private:
        arma::uword num_of_qubits = dw_numberOfQubits;
        arma::mat J;
        arma::vec h;
        arma::mat Q;
        double offset;

        void computeQUBO();
    public:
        arma::uword size() const {return num_of_qubits;}
        const arma::mat& get_J() const {return J;}
        const arma::vec& get_h() const {return h;}
        double get_offset() const {return offset;}
        const arma::mat& get_Q() const {return Q;}
        void read_file(const std::string&);

        Hamiltonian(arma::uword nQubits):
            num_of_qubits(nQubits){
            J = arma::zeros<arma::mat>(num_of_qubits,num_of_qubits);
            h = arma::zeros<arma::vec>(num_of_qubits);
            Q = arma::zeros<arma::mat>(num_of_qubits,num_of_qubits);
            offset = 0;
       };
        Hamiltonian():Hamiltonian(dw_numberOfQubits){};
        Hamiltonian(arma::vec in_h,arma::mat in_J):
            num_of_qubits(in_h.size()),J(in_J),h(in_h){
            computeQUBO();
        }
        Hamiltonian(const std::string& fileName, const arma::uword& nQubits):
            num_of_qubits(nQubits){
            read_file(fileName);
            computeQUBO();
        }

    }; //end class Hamiltonian

    class SimulatedAnnealing{
    private:
        arma::uword num_of_qubits;
        std::uniform_int_distribution<int> rand_qubit;
        Hamiltonian ham;
        BitString state; ///Initialises to zero
        double beta;

    public:
        SimulatedAnnealing() {
            num_of_qubits = dw_numberOfQubits;
            beta = DW_BETA;
            rand_qubit  = std::uniform_int_distribution<int>(0,num_of_qubits-1);
            ham         = Hamiltonian(num_of_qubits);
        };

        SimulatedAnnealing(const Hamiltonian& in_ham):
            num_of_qubits(in_ham.size()),ham(in_ham) {
            beta = DW_BETA;
            rand_qubit  = std::uniform_int_distribution<int>(0,num_of_qubits-1);
        };

        SimulatedAnnealing(const Hamiltonian& in_ham,
                           const BitString& in_state):
            num_of_qubits(in_ham.size()),ham(in_ham),state(in_state){
            beta = DW_BETA;
            rand_qubit  = std::uniform_int_distribution<int>(0,num_of_qubits-1);
        };

        SimulatedAnnealing(const Hamiltonian& in_ham, const BitString& in_state,
                           double in_beta):
            ham(in_ham), state(in_state), beta(in_beta) {
            rand_qubit  = std::uniform_int_distribution<int>(0,num_of_qubits-1);
        };

        double get_temperature() const {return 1/beta;}
        void set_temperature(double in_temperature){
            beta = 1/in_temperature;
        }

        double get_beta() const {return beta;}
        void set_beta(const double& in_beta){beta=in_beta;}

        BitString get_state() const {return state;}
        void set_state(const BitString& in_state){state=in_state;}

        double get_energy() const;
        void anneal();
    }; //end class Simulated Annealing

    class ParallelTempering{
    private:
        arma::uword num_of_instances=64;
        std::vector<std::unique_ptr<SimulatedAnnealing>> instances;
        Hamiltonian ham;
        arma::vec beta;
        arma::uword num_of_SA_anneal;
        arma::uword num_of_swaps;
        double base_beta = 1/DW_TEMPERATURE;
        double final_beta = 100.0;
        void common_init();
    public:
        ParallelTempering(const Hamiltonian&);
        ParallelTempering(const Hamiltonian&,arma::uword);
        ParallelTempering(const Hamiltonian&, const BitString&);
        ParallelTempering(const Hamiltonian&, const BitString&,
                          const arma::vec&);
        ParallelTempering(const Hamiltonian&, double, double);
        ~ParallelTempering();

        void set_num_of_SA_anneal(arma::uword num_anneal) {num_of_SA_anneal = num_anneal;}
        arma::uword get_num_of_SA_anneal() const {return num_of_SA_anneal;}

        void set_num_of_swaps(arma::uword num_swaps) {num_of_swaps = num_swaps;}
        arma::uword get_num_of_swaps() const {return num_of_swaps;}

    };//end class Parallel tempering;

} //end namespace pt.

/// Defining a overloaded operator for multiplying arma::mat to BitString.
template <class T>
const arma::Mat<T> operator* (const arma::Mat<T>&,const pt::BitString&);
///////////////////////////////////////////////////////////////////////////////////////////////


#endif //bitstring.hpp
