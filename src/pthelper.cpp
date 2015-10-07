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
#include <boost/circular_buffer.hpp>
#include <root/TGraph.h>
#include <root/TCanvas.h>
#include <root/TAxis.h>

pt::PTSave::PTSave(){
    //100 MB of buffer for compression.
    const std::streamsize compress_buff_size = 100 *1024*1024;

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
                                 ,arma::uword in_num_of_swaps,arma::uword in_num_of_qubits)
{
    num_of_instances = in_num_of_instances;
    num_of_anneals   = in_num_of_anneals;
    num_of_swaps     = in_num_of_swaps;
    num_of_qubits    = in_num_of_qubits;
    spin_overlap_array.set_size(num_of_instances,num_of_anneals*num_of_swaps+1);
    spin_overlap_mean.set_size(num_of_instances,num_of_anneals*num_of_swaps+1);
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

        spin_overlap_array(ii_instance,anneal_counter) =
            double(instance_overlap)/num_of_qubits;

        if (anneal_counter==0)
            spin_overlap_mean(ii_instance,0) = double(instance_overlap)/num_of_qubits;
        else
            spin_overlap_mean(ii_instance,anneal_counter)
                = (anneal_counter*spin_overlap_mean(ii_instance,anneal_counter-1)
                   +double(instance_overlap)/num_of_qubits)/(anneal_counter+1);
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

    const arma::uword max_plot_points =
        (num_of_swaps*num_of_anneals>2000)?2000:num_of_swaps*num_of_anneals;
    x = new Double_t[max_plot_points+1];
    y = new Double_t[max_plot_points+1];

    //We sample a finite number of points to keep graph clean.
    for (arma::uword ii=0;ii<=max_plot_points;ii++) {
        x[ii] = num_of_swaps*num_of_anneals/max_plot_points*ii; //Integer division.
        y[ii] = spin_overlap_mean(0,x[ii]); //Running average
    }

    std::cout << "y[0] = "<<y[0]<<" x[0] = "<<x[0]<<std::endl;
    TCanvas* canvas = new TCanvas("haha");
    TGraph* gr = new TGraph(max_plot_points+1,x,y);
    canvas->SetLogy(1);
    gr->Draw("AP");
    gr->GetXaxis()->SetTitle("N_{mcs}");
    gr->GetXaxis()->CenterLabels();
    gr->GetYaxis()->SetTitle("Spin overlap mean <q>");
    gr->GetYaxis()->CenterLabels();
    canvas->Print(filename.c_str());
    delete gr;
    delete canvas;

    //Write these samples to disk.
    arma::mat writeMatrix(max_plot_points+1,2);
    writeMatrix.col(0) = arma::mat(x,max_plot_points+1,1);
    writeMatrix.col(1) = arma::mat(y,max_plot_points+1,1);
    writeMatrix.save(filename+"data.txt",arma::raw_ascii);

    delete[] x; delete[] y;
}

bool pt::PTTestThermalise::has_thermalised(){
    return flag_thermalised;
}

pt::PTTestThermalise::PTTestThermalise
(arma::uword in_num_of_qubits,arma::uword in_instance_number,double in_tolerance){
    anneal_counter      = 0 ;
    num_of_qubits       = in_num_of_qubits;
    instance_number     = in_instance_number;
    tolerance           = in_tolerance;
    flag_thermalised    = false;

    mean_overlap_buffer = boost::circular_buffer<double>(5);
    for(int ii=0;ii<5;ii++)
        mean_overlap_buffer.push_back(0);
}

void pt::PTTestThermalise::compute(const std::vector<std::unique_ptr<pt::boost_bitset>>& ins1,
                     const std::vector<std::unique_ptr<pt::boost_bitset>>& ins2,
                                  const arma::vec& en1, const arma::vec& en2){
    if(flag_thermalised) //do nothing if already thermal.
        return;

    //Keep track of last five value of mean overlap and calculate the differential.
    static double last_mean_value = 0;
    boost_bitset overlap_bitset = ~((*ins1[instance_number])^(*ins2[instance_number]));
    auto block_vector = boost_bitset_to_vector(overlap_bitset);

    double instance_overlap=0;
    for (auto& ii:block_vector)
        instance_overlap += num_of_1bits_fast(ii);
    instance_overlap/=num_of_qubits;

    double current_mean_value =
        (anneal_counter*last_mean_value + instance_overlap)/(anneal_counter+1);
    mean_overlap_buffer.push_back(current_mean_value);

    const double* buffer_data = mean_overlap_buffer.linearize();
    arma::vec values(buffer_data,5);
    double derivative_mean_overlap = pt::fourth_order_derivative(values);

    if ((anneal_counter>5) && (std::abs(derivative_mean_overlap) < tolerance)){
        flag_thermalised = true;
    }

    last_mean_value  = current_mean_value;
    anneal_counter++;
}
