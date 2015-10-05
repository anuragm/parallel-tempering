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
#include <root/TGraph.h>
#include <root/TCanvas.h>
#include <root/TAxis.h>

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

pt::PTSpinOverlap::PTSpinOverlap(arma::uword in_num_of_instances, arma::uword in_num_of_anneals
                                 ,arma::uword in_num_of_swaps)
{
    num_of_instances = in_num_of_instances;
    num_of_anneals   = in_num_of_anneals;
    num_of_swaps     = in_num_of_swaps;
    spin_overlap_array.set_size(num_of_instances,num_of_anneals*num_of_swaps+1);
    anneal_counter=0;
}

void pt::PTSpinOverlap::compute(const std::vector<std::unique_ptr<boost_bitset>>& ins1,
                                const std::vector<std::unique_ptr<boost_bitset>>& ins2,
                                const arma::vec& en1, const arma::vec& en2){
    for(arma::uword ii_instance=0;ii_instance<num_of_instances;ii_instance++){
        boost_bitset overlap_bitset = ~((*ins1[ii_instance])^(*ins2[ii_instance]));
        auto block_vector = boost_bitset_to_vector(overlap_bitset);
        arma::uword instance_overlap=0;
        for (auto& ii:block_vector)
            instance_overlap += num_of_1bits_fast(ii);
        spin_overlap_array(ii_instance,anneal_counter) = instance_overlap;
    }
    anneal_counter++;
}

void pt::PTSpinOverlap::flush_to_files(std::string filename){
    std::ofstream file_obj(filename,std::ios::binary);
    compress_overlap.push(boost::iostreams::zlib_compressor(compression_level));
    compress_overlap.push(file_obj);

    compress_overlap.write(reinterpret_cast<const char*>(spin_overlap_array.memptr()),
                           sizeof(unsigned int)*spin_overlap_array.n_elem);
    compress_overlap.pop();
    file_obj.close();
}

void pt::PTSpinOverlap::plot_to_file_overlap_mean(std::string filename){
    //We want to plot mean overlap <q>, averages over anneal steps.
    Double_t* x; Double_t* y;
    arma::vec overlap_mean(num_of_swaps);

    //First, we average over the number of anneals, thus over time step is one single exchange
    for(arma::uword ii_swap=0;ii_swap<num_of_swaps;ii_swap++){
        overlap_mean(ii_swap) = 1.0/num_of_anneals *
            arma::accu(spin_overlap_array.row(0).
                        subvec(ii_swap*num_of_anneals,(ii_swap+1)*num_of_anneals-1));
    }

    const arma::uword max_plot_points = (num_of_swaps>1000)?1000:num_of_swaps;
    x = new Double_t[max_plot_points+1];
    y = new Double_t[max_plot_points+1];
    x[0] = 0;
    y[0] = overlap_mean(0);
    //Then we take cumulative average to get average <q> for a PT algorithm running for swaps
    //'n'
    for (arma::uword ii=1;ii<max_plot_points;ii++) {
        x[ii] = num_of_swaps/max_plot_points*ii; //Integer division.
        y[ii] = arma::sum(overlap_mean.subvec(0,x[ii]))/x[ii]; //Running average
    }
    TCanvas* canvas = new TCanvas("haha");
    TGraph* gr = new TGraph(max_plot_points,x,y);
    canvas->SetLogy(1);
    gr->Draw("AP");
    gr->GetXaxis()->SetTitle("Swap number");
    gr->GetYaxis()->SetTitle("Spin overlap mean <q>");
    canvas->Print(filename.c_str());
    delete gr;
    delete canvas;
    delete[] x; delete[] y;
}
