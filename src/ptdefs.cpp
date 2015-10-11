#include "ptdefs.hpp"
#include <armadillo>

std::vector<pt::defaultBlock> pt::boost_bitset_to_vector(const pt::boost_bitset& state){
    arma::uword vector_size = state.size()/sizeof(pt::defaultBlock)/CHAR_BIT;
    if(vector_size==0)
        vector_size=1;
    std::vector<pt::defaultBlock> conv_vector(vector_size);
    boost::to_block_range(state, conv_vector.begin());
    return conv_vector;
}

//Returns f'(x_2) when given a list of numbers x_0, x_1, x_2, x_3 and x_4 in an array.
double pt::fourth_order_derivative(arma::Col<double> values){
    //Values taken from
    //https://en.wikipedia.org/wiki/Finite_difference_coefficient#Central_finite_difference.
    arma::rowvec coeffs = {1.0/12,-2.0/3,0,2.0/3,-1.0/12};
    return arma::as_scalar(coeffs*values);
}

//Returns f'(x_0) when given a list of numbers x_0,x_1,x_2,x_3 and x_4 in array.
double pt::forward_derivative(arma::Col<double> values){
    arma::rowvec coeffs = {-25.0/12,4,-3,4.0/3,-1.0/4};
    return arma::as_scalar(coeffs*values);
}

