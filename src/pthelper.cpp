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
#include <gnuplot-iostream.h>

/////////////////////////////////////////////////////////////////////////////////
/// PT::PTSave
/////////////////////////////////////////////////////////////////////////////////

pt::PTSave::PTSave(){
    //100 MB of buffer for compression.
    const std::streamsize compress_buff_size = 100*1024*1024;

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

//////////////////////////////////////////////////////////////////////////////////////////////
/// PT:PTSpinOverlap
/////////////////////////////////////////////////////////////////////////////////////////////

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

    const arma::uword max_plot_points =
        (num_of_swaps*num_of_anneals>2000)?2000:num_of_swaps*num_of_anneals;
    arma::mat plot_points(max_plot_points,2,arma::fill::zeros);
    plot_points(0,0) = 0; plot_points(0,1) = spin_overlap_mean(0,0);

    //We sample a finite number of points to keep graph clean.
    for (arma::uword ii=1;ii<(max_plot_points);ii++) {
        double x_point = num_of_swaps*num_of_anneals/max_plot_points*ii; //Integer division.
        double y_point = spin_overlap_mean(0,x_point);
        plot_points(ii,0) = x_point;
        plot_points(ii,1) = y_point;
    }

    std::cout << "y[0] = "<<plot_points(0,1)<<" x[0] = "<<plot_points(0,0)<<std::endl;

    Gnuplot gp;
    double y_max_lim = 1.01*arma::max(plot_points.col(1));
    double y_min_lim = 0.99*arma::min(plot_points.col(1));
    std::cout << "Minimum overlap value is "<<y_min_lim<<" and maximum overlap values is "
              <<y_max_lim<<std::endl;
    plot_points.save("overlap_points.txt",arma::raw_ascii);
    std::cin.get();
    gp<<"set yrange ["<<y_min_lim<<":"<<y_max_lim<<"]\n";
    //gp<<"set logscale y\n";
    gp<<"plot "<<gp.binFile1d(plot_points,"record")<<"title 'spin overlap'\n";
}

////////////////////////////////////////////////////////////////////////////////////
/// PT:PTTestThermalise
///////////////////////////////////////////////////////////////////////////////////

bool pt::PTTestThermalise::has_thermalised(){
    bool is_thermal = true;
    //If any of the instances hasn't thermalised, return false.
    for(unsigned ii_instance=0;ii_instance<num_of_instances;ii_instance++){
        is_thermal = (is_thermal & flag_thermalised[ii_instance]);
    }
    return is_thermal;
}

pt::PTTestThermalise::PTTestThermalise
(arma::uword in_num_of_qubits,arma::uword in_num_of_instances,double in_tolerance):
    num_of_qubits(in_num_of_qubits),
    num_of_instances(in_num_of_instances),
    mean_overlap_buffer(num_of_instances,pt::FixedQueue<double>(5)),
    tolerance(in_tolerance){

    anneal_counter      = 0 ;
    flag_thermalised    = std::vector<bool>(num_of_instances,false);

    for(auto& buff_instance : mean_overlap_buffer)
        for(int ii=0;ii<5;ii++)
            buff_instance.push_back(0);
}

void pt::PTTestThermalise::compute(const std::vector<std::unique_ptr<pt::boost_bitset>>& ins1,
                     const std::vector<std::unique_ptr<pt::boost_bitset>>& ins2,
                                  const arma::vec& en1, const arma::vec& en2){
    if(has_thermalised()) //do nothing if already thermal.
        return;

    static std::vector<double> last_mean_value(num_of_instances,0);
    //Keep track of last five value of mean overlap and calculate the differential.
    for(arma::uword ii_instance=0;ii_instance<num_of_instances;ii_instance++){
        boost_bitset overlap_bitset = ~((*ins1[ii_instance])^(*ins2[ii_instance]));
        auto block_vector = boost_bitset_to_vector(overlap_bitset);

        double instance_overlap=0;
        for (auto& ii:block_vector)
            instance_overlap += num_of_1bits_fast(ii);
        instance_overlap/=num_of_qubits;

        double current_mean_value =
            (anneal_counter*last_mean_value[ii_instance] + instance_overlap)/(anneal_counter+1);
        mean_overlap_buffer[ii_instance].push_back(current_mean_value);

        const double* buffer_data = mean_overlap_buffer[ii_instance].data();
        arma::vec values(buffer_data,5);
        double derivative_mean_overlap = pt::fourth_order_derivative(values);

        if ((anneal_counter>5) && (std::abs(derivative_mean_overlap) < tolerance)){
            flag_thermalised[ii_instance] = true;
        }
        else
            flag_thermalised[ii_instance] = false; //Reflag to false if any of the instance
                                                   //shows not thermal state.

        last_mean_value[ii_instance]  = current_mean_value;
    }
    anneal_counter++;
}

//////////////////////////////////////////////////////////////////////////////
/// pt::PTAutocorrelation
/////////////////////////////////////////////////////////////////////////////


pt::PTAutocorrelation::PTAutocorrelation
(arma::uword in_num_of_anneals, arma::uword in_num_of_instances){
    num_of_instance = in_num_of_instances;
    total_anneals = in_num_of_anneals;
    energies1.set_size(num_of_instance,total_anneals);
    anneal_counter = 0;
}

void pt::PTAutocorrelation::compute(const std::vector<std::unique_ptr<pt::boost_bitset>>& ins1,
                                    const std::vector<std::unique_ptr<pt::boost_bitset>>& ins2,
                                    const arma::vec& en1, const arma::vec& en2){
    energies1.col(anneal_counter) = en1;
    anneal_counter++;
}

arma::uword pt::PTAutocorrelation::get_correlation_length(){
    //Perform the binning test to get the correlation length.
    //Bin in sizes 1,2,4,16,....
    arma::uword num_of_bins = arma::uword(std::log2(total_anneals));
    arma::mat en1_variance(num_of_instance,num_of_bins);

    for(arma::uword ii_bin=0;ii_bin<num_of_bins;ii_bin++){
        arma::uword bin_size = (1<<ii_bin);
        arma::mat data_series_en1(num_of_instance,total_anneals/bin_size);
        for(arma::uword ii=0;ii<(total_anneals/bin_size);ii++){
            data_series_en1.col(ii) =
                arma::sum(energies1.cols(ii*bin_size,(ii+1)*bin_size-1),1)/bin_size;
        }
        en1_variance.col(ii_bin) = arma::var(data_series_en1,1,1);
    }

    //Now, we need to find out where the variance stablises first.
    arma::uvec corr_length(num_of_instance,arma::fill::zeros);
    arma::vec five_elements;
    double variance_derivative;
    for(arma::uword ii_instance=0;ii_instance<num_of_instance;ii_instance++){
        for(arma::uword ii_bin=0;ii_bin<(num_of_bins-4);ii_bin++){
            five_elements    = en1_variance.row(ii_instance).cols(ii_bin,ii_bin+4).t();
            variance_derivative = pt::forward_derivative(five_elements);
            if(variance_derivative<1e-7){
                corr_length(ii_instance)=(1<<ii_bin);
                break;
            }
        }
    }

    return arma::max(corr_length);
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// PT::PTStore
///////////////////////////////////////////////////////////////////////////////////////////////

pt::PTStore::PTStore(arma::uword in_corr_length,arma::uword in_anneals,
                     arma::uword in_instances,arma::uword in_qubits):
    correlation_length(in_corr_length),total_anneals(in_anneals),
    num_of_instances(in_instances),num_of_qubits(in_qubits){
    energies.set_size(in_instances,total_anneals/correlation_length);
    anneal_counter = 0;
}

void pt::PTStore::compute(const std::vector<std::unique_ptr<pt::boost_bitset>>& ins1,
                          const std::vector<std::unique_ptr<pt::boost_bitset>>& ins2,
                          const arma::vec& en1, const arma::vec& en2){
    if(anneal_counter%correlation_length ==0){
        arma::uword array_loc = (anneal_counter/correlation_length);
        energies.col(array_loc) = en1;
    }
    anneal_counter++;
}

arma::mat pt::PTStore::get_energies() const{
    return energies;
}
