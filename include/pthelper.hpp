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

}
#endif
