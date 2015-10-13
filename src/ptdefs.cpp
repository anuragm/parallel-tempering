#include "ptdefs.hpp"
#include <armadillo>


//Initialise global random number generators.
namespace pt{
    const auto time_seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 rand_eng(time_seed);
    std::uniform_real_distribution<double> uniform_dist =
        std::uniform_real_distribution<double>(0.0,1.0);
}

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

pt::OverlapHistogram::OverlapHistogram(unsigned int in_num_bins):
    num_of_bins(in_num_bins/2*2),count(num_of_bins,0){
    if(in_num_bins%2!=0){
        std::cout << "Number of bins is odd. Adjusted to nearest even size "
                  <<num_of_bins;
    }
    offset = num_of_bins/2;
}

bool pt::OverlapHistogram::is_symmetric(double tolerance) const{
    //unsigned int total_count = std::accumulate(count.begin(),count.end(),0);
    //Check if each P(q) == P(-q)
    for(unsigned int ii=0;ii<(num_of_bins/2);ii++){
        if(count[num_of_bins-1-ii]==0){
            std::cout << "Insufficient statistics ";
            return false;
        }
        if(std::abs((1.0*count[ii]/count[num_of_bins-1-ii])-1)<tolerance)
            return false;
    }
    return true;
}

void pt::OverlapHistogram::push_value(double in_value){
    assert(in_value<=1.0 && in_value>=-1.0);
    unsigned int location;
    if(in_value==1.0) //Can do double comparison as anything less than 1.0 is OK.
        location = num_of_bins-1;
    else
        location = static_cast<int>(std::floor(in_value*offset))+offset;

    count[location]++;
}

arma::mat pt::OverlapHistogram::get_histogram() const{
    arma::uword total_counts = std::accumulate(count.begin(),count.end(),0);
    arma::mat hist(num_of_bins,2);
    for(arma::uword ii_bin=0;ii_bin<num_of_bins;ii_bin++){
        hist(ii_bin,0) = 2.0*(int(ii_bin)-int(offset))/num_of_bins;
        hist(ii_bin,1) = 1.0*count[ii_bin]/total_counts;
    }

    return hist;
}
