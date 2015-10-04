/**
 *   \file pt.hpp
 *   \brief Interface for the various classes used in the parallel
 *   tempering algorithm.
 *
 *   All definitions in this interface belong to the namespace
 *   pt. Different classes provide for different functions.
 *
 */

#ifndef PT_HPP
#define PT_HPP 1

#include <armadillo>
#include <random>
#include <chrono>
#include <cmath>
#include <memory>
#include <boost/dynamic_bitset.hpp>
#include "ptdefs.hpp"
#include "pthelper.hpp"

namespace pt{

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
        double flip_energy_diff(int, const boost_bitset&);
        double energy_diff(boost_bitset, const boost_bitset&);
        double get_energy(const boost_bitset&) const;
        void scale_to_unit();

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

    enum instance_number {INSTANCES_1, INSTANCES_2};

    class ParallelTempering{
    private:
        arma::uword num_of_instances;
        Hamiltonian ham;
        arma::vec beta;
        arma::uword num_of_SA_anneal;
        arma::uword num_of_swaps;
        double base_beta;
        double final_beta;
        arma::uword anneal_counter;
        std::uniform_int_distribution<int> rand_qubit;

        bool flag_init;

        std::vector<std::unique_ptr<boost_bitset>> instances1;
        std::vector<std::unique_ptr<boost_bitset>> instances2;
        arma::vec energies1;
        arma::vec energies2;

        std::vector<PTHelper*> helper_objects;
        void init();

    public:
        ParallelTempering() = delete;
        ParallelTempering(const Hamiltonian& in_ham, arma::uword in_instances=64);
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

        arma::vec get_beta() const {return beta;}

        void perform_anneal(){
            perform_anneal(num_of_SA_anneal);
        }

        void push(PTHelper* pt_help_obj){
            helper_objects.push_back(pt_help_obj);
        }

        void perform_anneal(arma::uword anneal_steps);
        void perform_swap();
        void run();
        arma::vec get_energies(instance_number) const;
    };//end class Parallel tempering;
} //end namespace pt.

//Helper file needs to be included at end so as to ensure that the included file can see all
//the definitions of this file hidden by header guard while recursive includes. This can be
//solved by having a skeleton include meta-file, the way boost does it.

#endif //bitstring.hpp
