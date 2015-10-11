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
#include <type_traits>

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
        if(!std::is_integral<T>::value)
            throw std::logic_error("Cannot find bits for non integer type\n");

        unsigned int c;
        v = v - ((v >> 1) & (T)~(T)0/3);                           // temp
        v = (v & (T)~(T)0/15*3) + ((v >> 2) & (T)~(T)0/15*3);      // temp
        v = (v + (v >> 4)) & (T)~(T)0/255*15;                      // temp
        c = (T)(v * ((T)~(T)0/255)) >> (sizeof(T) - 1) * CHAR_BIT; // count
        return c;
    }

    double fourth_order_derivative(arma::Col<double> values);
    double forward_derivative(arma::Col<double> values);

    template<typename T>
    class FixedQueue{
    private:
        T* internal_array;
        std::size_t array_size;
        std::size_t element_location;
    public:
        std::size_t size() const{
            return array_size;
        }

        FixedQueue(std::size_t in_array_size):array_size(in_array_size){
            internal_array = new T[array_size];
            element_location = 0;
        }

        FixedQueue(const FixedQueue& q){
            array_size = q.array_size;
            element_location = q.element_location;
            internal_array = new T[array_size];
            std::copy(q.internal_array,q.internal_array+q.array_size,internal_array);
        }

        ~FixedQueue(){
            delete[] internal_array;
        }

        void push_back(T value){
            internal_array[element_location%array_size] = value;
            element_location++;
        }

        T* data() const{
            return internal_array;
        }

        std::vector<T> data_vector() const{
            std::vector<T> temp_vector(array_size);
            std::copy(internal_array,internal_array+array_size,temp_vector.begin());
            return temp_vector;
        }
    };
}

#endif
