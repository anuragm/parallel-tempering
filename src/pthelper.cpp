/**
 *   \file pthelper.cpp
 *   \brief Implements pthelper.hpp
 *
 *  The pt helper interface defines one interface that can be implemented to hook arbitrary
 *  computation to the parallel tempering algorithm. We include an interface, and sample
 *  implementations which save all the states and energies during computation to disk, as well
 *  as a class that computes spin overlap and estimates when the parallel tempering algorithm
 *  has thermalised.
 *
 */

#include "pthelper.hpp"
#include <armadillo>
#include <boost/iostreams/device/file.hpp> //include to read-write files.
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/zlib.hpp>

pt::PTSave::PTSave(){
    const std::streamsize compress_buff_size = 100 *1024*1024; //100 MB of buffer for compression.
    
    //Initialise the string streams to take binary output
    instance_buffer1 = std::ostringstream(std::stringstream::binary);
    instance_buffer2 = std::ostringstream(std::stringstream::binary);
    energies_buffer1 = std::ostringstream(std::stringstream::binary);
    energies_buffer2 = std::ostringstream(std::stringstream::binary);

    //Initialise the filtering ostreams.
    compress_ins1.push(boost::iostreams::zlib_compressor
                       (compression_level,compress_buff_size));
    compress_ins1.push(instance_buffer1);

    compress_ins2.push(boost::iostreams::zlib_compressor
                       (compression_level,compress_buff_size));
    compress_ins2.push(instance_buffer2);

    compress_en1.push(boost::iostreams::zlib_compressor
                      (compression_level,compress_buff_size));
    compress_en1.push(energies_buffer1);

    compress_en2.push(boost::iostreams::zlib_compressor
                      (compression_level,compress_buff_size));
    compress_en2.push(energies_buffer2);
}

void pt::PTSave::compute(const std::vector<std::unique_ptr<boost_bitset>>& instance1,
                         const std::vector<std::unique_ptr<boost_bitset>>& instance2,
                         const arma::vec& energies1, const arma::vec& energies2){

    //Write the energies to compressed buffer.
    compress_en1.write(reinterpret_cast<const char*>(energies1.memptr()),
                       sizeof(double)*energies1.n_elem);
    compress_en2.write(reinterpret_cast<const char*>(energies2.memptr()),
                       sizeof(double)*energies2.n_elem);

    //Convert the instances to a vector, and write it to buffer.
    for(auto& state: instance1){
        auto block_vector = pt::boost_bitset_to_vector(*state);
        compress_ins1.write(reinterpret_cast<const char*>(block_vector.data()),
                            sizeof(pt::defaultBlock)*block_vector.size());
        };
    for(auto& state: instance2){
        auto block_vector = pt::boost_bitset_to_vector(*state);
        compress_ins2.write(reinterpret_cast<const char*>(block_vector.data()),
                            sizeof(pt::defaultBlock)*block_vector.size());
    };
}

void pt::PTSave::flush_to_files(std::string file_prefix){
    //Write the energies to corresponding files.
    //Pop compression filters to force write to ostringstreams.
    compress_ins1.pop(); compress_ins2.pop(); compress_en1.pop(); compress_en2.pop();

    std::ofstream out_file;
    std::string temp_str;

    auto write_to_file_ostream = [&] (const std::string& file_name,
                                      std::ostringstream& out_stream){
        temp_str = out_stream.str();
        out_file.open(file_name,std::ios::binary);
        out_file.write(temp_str.c_str(),temp_str.size()-1); //Don't write \0 to file.
        out_file.close();
    };
    write_to_file_ostream(file_prefix+".1.energies",energies_buffer1);
    write_to_file_ostream(file_prefix+".2.energies",energies_buffer2);
    write_to_file_ostream(file_prefix+".1.states",instance_buffer1);
    write_to_file_ostream(file_prefix+".2.states",instance_buffer2);
}
