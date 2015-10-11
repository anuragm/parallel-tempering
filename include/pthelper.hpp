/**
 *   \file pthelper.hpp
 *   \brief Header file to include helper functions for parallel tempering
 *
 *  While parallel tempering, we need various helper classes which compute different things
 *  depending on the requirement of main algorithm. I would define appropriate interfaces here.
 *
 */
#ifndef PTHELPER_HPP
#define PTHELPER_HPP 1

#include "ptdefs.hpp"
#include <iostream>
#include <boost/iostreams/device/file.hpp> //include to read-write files.
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/zlib.hpp>
#include <boost/circular_buffer.hpp>
#include <armadillo>

namespace pt{
    class PTHelper{
    public:
        virtual void compute(const std::vector<std::unique_ptr<pt::boost_bitset>>&,
                             const std::vector<std::unique_ptr<pt::boost_bitset>>&,
                             const arma::vec&, const arma::vec&) = 0;
    };

    class PTSave : public PTHelper{
    private:
        std::ostringstream instance_buffer1, instance_buffer2;
        std::ostringstream energies_buffer1, energies_buffer2;
        boost::iostreams::filtering_ostream
        compress_ins1, compress_ins2, compress_en1, compress_en2;

        unsigned compression_level=6;
    public:
        PTSave();
        void compute(const std::vector<std::unique_ptr<pt::boost_bitset>>&,
                     const std::vector<std::unique_ptr<pt::boost_bitset>>&,
                     const arma::vec&, const arma::vec&);
        void flush_to_files(std::string);
    };

    class PTSpinOverlap: public PTHelper{
    private:
        arma::Mat<double> spin_overlap_array;
        arma::Mat<double> spin_overlap_mean;
        boost::iostreams::filtering_ostream compress_overlap;
        arma::uword num_of_instances, num_of_anneals, num_of_swaps;
        arma::uword num_of_qubits;

        arma::uword anneal_counter;

        unsigned compression_level=6;
    public:
        PTSpinOverlap() = delete;
        PTSpinOverlap(arma::uword,arma::uword,arma::uword,arma::uword);
        void compute(const std::vector<std::unique_ptr<pt::boost_bitset>>&,
                     const std::vector<std::unique_ptr<pt::boost_bitset>>&,
                     const arma::vec&, const arma::vec&);
        void flush_to_files(std::string);
        void plot_to_file_overlap_mean(std::string);
    };

    class PTTestThermalise : public PTHelper{
    private:
        std::vector<bool> flag_thermalised;
        arma::uword anneal_counter;
        arma::uword num_of_qubits;
        arma::uword num_of_instances;
        std::vector<pt::FixedQueue<double>> mean_overlap_buffer;
        double tolerance;
    public:
        PTTestThermalise() = delete;
        PTTestThermalise(arma::uword,arma::uword,double in_tolerance=1e-8);
        void compute(const std::vector<std::unique_ptr<pt::boost_bitset>>&,
                     const std::vector<std::unique_ptr<pt::boost_bitset>>&,
                     const arma::vec&, const arma::vec&);
        bool has_thermalised();
    };

    class PTAutocorrelation : public PTHelper{
    private:
        arma::uword total_anneals;
        arma::uword anneal_counter;
        arma::uword num_of_instance;
        arma::mat energies1;
    public:
        void compute(const std::vector<std::unique_ptr<pt::boost_bitset>>&,
                     const std::vector<std::unique_ptr<pt::boost_bitset>>&,
                     const arma::vec&, const arma::vec&);
        arma::uword get_correlation_length();
        PTAutocorrelation(arma::uword,arma::uword);
        PTAutocorrelation() = delete;
    };

    class PTStore : public PTHelper{
    private:
        arma::uword correlation_length;
        arma::uword total_anneals;
        arma::uword num_of_instances;
        arma::uword anneal_counter;
        arma::uword num_of_qubits;
        arma::mat energies;
    public:
        PTStore(arma::uword,arma::uword,arma::uword,arma::uword);
        void compute(const std::vector<std::unique_ptr<pt::boost_bitset>>&,
                     const std::vector<std::unique_ptr<pt::boost_bitset>>&,
                     const arma::vec&, const arma::vec&);
        arma::mat get_energies() const;
    };
} //end of namespace pt.
#endif
