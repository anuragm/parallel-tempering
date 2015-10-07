/**
 *   \file ptdefs.hpp
 *   \brief Defines common constants and typedefs for parallel tempering algorithm.
 *
 */

#ifndef PTDEFS_HPP
#define PTDEFS_HPP 1

#include <iostream>
#include <random>
#include <boost/dynamic_bitset.hpp>
#include <armadillo>

namespace pt{
    using defaultBlock = unsigned long ;
    using boost_bitset = boost::dynamic_bitset<defaultBlock>;
    //Global variables
    const double DW_TEMPERATURE=0.10991;
    const double DW_BETA = 1/DW_TEMPERATURE;
    const arma::uword DW_NUM_OF_QUBIT=512;

    const auto time_seed = std::chrono::system_clock::now().time_since_epoch().count();
    extern std::mt19937 rand_eng;
    extern std::uniform_real_distribution<double> uniform_dist;

    std::vector<defaultBlock> boost_bitset_to_vector(const boost_bitset&);

    //Function to count number of 1's in bits of unsigned integers of length up-to 128 bits.
    //Code taken from http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
    template<typename T>
    unsigned int num_of_1bits_fast(T v){
        unsigned int c;
        v = v - ((v >> 1) & (T)~(T)0/3);                           // temp
        v = (v & (T)~(T)0/15*3) + ((v >> 2) & (T)~(T)0/15*3);      // temp
        v = (v + (v >> 4)) & (T)~(T)0/255*15;                      // temp
        c = (T)(v * ((T)~(T)0/255)) >> (sizeof(T) - 1) * CHAR_BIT; // count
        return c;
    }

    double fourth_order_derivative(arma::Col<double> values);

}

#endif
