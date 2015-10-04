#include "ptdefs.hpp"

std::vector<pt::defaultBlock> pt::boost_bitset_to_vector(const pt::boost_bitset& state){
    std::vector<pt::defaultBlock> conv_vector
        (state.size()/sizeof(pt::defaultBlock)/CHAR_BIT);
    boost::to_block_range(state, conv_vector.begin());
    return conv_vector;
}
