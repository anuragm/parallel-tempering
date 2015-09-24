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
#define BITSTRING_HPP 1

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
    const arma::uword DW_NUM_OF_QUBIT=512;

    const auto time_seed = std::chrono::system_clock::now().time_since_epoch().count();
    extern std::mt19937 rand_eng;
    extern std::uniform_real_distribution<double> uniform_dist;

    class Hamiltonian{
    private:
        arma::uword num_of_qubits = DW_NUM_OF_QUBIT;
        arma::mat J;
        arma::vec h;
        arma::mat Q;
        std::vector<arma::uvec> neighbours;
        double offset;
        void pre_compute();

    public:
        arma::uword size() const {return num_of_qubits;}
        const arma::mat& get_J() const {return J;}
        const arma::vec& get_h() const {return h;}
        double get_offset() const {return offset;}
        const arma::mat& get_Q() const {return Q;}
        void read_file(const std::string&, int offset);
        const arma::uvec& get_ngbrs(int qubit_number) const{
            return neighbours[qubit_number];
        };
        double flip_energy_diff(int, const boost::dynamic_bitset<>&);
        double energy_diff(boost::dynamic_bitset<>, const boost::dynamic_bitset<>&);
        double get_energy(const boost::dynamic_bitset<>&) const;

        Hamiltonian(arma::uword nQubits):
            num_of_qubits(nQubits){
            J = arma::zeros<arma::mat>(num_of_qubits,num_of_qubits);
            h = arma::zeros<arma::vec>(num_of_qubits);
            Q = arma::zeros<arma::mat>(num_of_qubits,num_of_qubits);
            offset = 0;
       };
        Hamiltonian():Hamiltonian(DW_NUM_OF_QUBIT){};
        Hamiltonian(arma::vec in_h,arma::mat in_J):
            num_of_qubits(in_h.size()),J(in_J),h(in_h){
            pre_compute();
        }
        Hamiltonian(const std::string& fileName, const arma::uword& nQubits, int offset=0):
            num_of_qubits(nQubits){
            read_file(fileName,offset);
            pre_compute();
        }

    }; //end class Hamiltonian

    class ParallelTempering{
    private:
        arma::uword num_of_instances=64;
        std::vector<std::unique_ptr<boost::dynamic_bitset<>>> instances;
        Hamiltonian ham;
        arma::vec beta;
        arma::uword num_of_SA_anneal;
        arma::uword num_of_swaps;
        double base_beta = DW_BETA;
        double final_beta = DW_BETA/10;
        std::uniform_int_distribution<int> rand_qubit;
        void common_init();
    public:
        ParallelTempering(const Hamiltonian&);
        ParallelTempering(const Hamiltonian&, arma::uword);
        ParallelTempering(const Hamiltonian&, const arma::vec&);
        ParallelTempering(const Hamiltonian&, double, double,arma::uword);
        ~ParallelTempering();

        void set_num_of_SA_anneal(arma::uword num_anneal) {
            num_of_SA_anneal = num_anneal;
        }

        arma::uword get_num_of_SA_anneal() const {
            return num_of_SA_anneal;
        }

        void set_num_of_swaps(arma::uword num_swaps) {
            num_of_swaps = num_swaps;
        }

        arma::uword get_num_of_swaps() const {
            return num_of_swaps;
        }

        void perform_anneal(){
            perform_anneal(num_of_SA_anneal);
        }
        void perform_anneal(arma::uword anneal_steps);
        void perform_swap();
        void run();
        arma::vec get_energies() const;
    };//end class Parallel tempering;
} //end namespace pt.

#endif //bitstring.hpp
