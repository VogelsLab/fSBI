#include "auryn.h"
#include "T4wvceciMLPConnection.h"
#include "CVAIFGroup.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>

/*!\file 
* This is a part of Synapseek, written by Basile Confavreux. It uses the spiking network simulator Auryn written by Friedemann Zenke.
 * This file is intended to be called by the corresponding python synapseek innerloop as part of a meta-optimization of plasticity rules.
 * It simulates a E-I spiking network with I-E plasticity, parametrized with an MLP (check T4wvceciMLPConnection for more info).
 * Implementing simulation protocol similar From Vogels et al 2011, inhibitory plasticity. Adapted from sim_isp_orig, written by Friedemann Zenke.
 * Simulates and score a network of COBA Exc and Inh neurons, with AMPA and NMADA exc conductances, and inhibitory plasticity
 * Desired behaviour: reach and maintain target firing rate
 * Loss: <(firing rate - target)**2/(firing rate + 0.1)>neurons, time
 * */

namespace po = boost::program_options;
using namespace auryn;

std::vector<float> parse_input_plasticity(std::string s1, int expected_rule_size)
{
	std::string s = s1;
	// parsing the command line rule argument: (it is given as a string because multitoken() is bugged: negative numbers cause errors)
	std::vector<float> rule_aux(expected_rule_size);
	for (int i = 0; i < rule_aux.size(); i++){
		rule_aux[i] = -1.;
	}
	std::string delimiter = "a";
	size_t pos = 0;
	int ct = 0;
	std::string token;

	token = s.substr(0, pos); // remove the first a (needed in case first param is negative, or is it?)
	s.erase(0, pos + delimiter.length());

	while ((pos = s.find(delimiter)) != std::string::npos) { // parse the rest of the expression
		token = s.substr(0, pos);
		rule_aux[ct] = boost::lexical_cast<float>(token);
		s.erase(0, pos + delimiter.length());
		ct ++;
	}
	return(rule_aux);
}

int main(int ac, char* av[]) 
{
	/////////////////////////////////////////////////////////
	// Get simulation parameters from command line options //
	/////////////////////////////////////////////////////////

	std::string ID = "0";
	int NE;
	int NI;

	float tau_ampa = 0;
	float tau_gaba = 0;
	float tau_nmda = 0;
	float ampa_nmda_ratio = 0;

	float tau_vtrace = 1.;
	float tau_cexc = 1.;
	float tau_cinh = 1.;

	float wee;
	float wei;
	float wie;
	float wii;
	std::vector<float> rule_agg(6);
	std::vector<float> rule_EE(6);
	std::vector<float> rule_IE(6);
	std::string rule_str;
	int nh1;
	int nh2;
	float wmax;
	float sparseness;
	float length_no_scoring;
	float length_scoring;
	int N_inputs;
	float sparseness_poisson;
	float w_poisson;
	float poisson_rate;
	AurynFloat min_rate_checker;
	AurynFloat max_rate_checker;
	AurynFloat tau_checker;
	std::string workdir;

	int n_recorded = 1000;
	bool record_i = false;
	int n_recorded_i = 500;

    try {
        po::options_description desc("Allowed options");
        desc.add_options()
			("ID", po::value<std:: string>(), "ID to name the monitor output files correctly")
			("NE", po::value<int>(), "number of excitatory neurons")
			("NI", po::value<int>(), "number of inhibitory neurons")
			("tau_ampa", po::value<float>(), "tau_ampa")
			("tau_gaba", po::value<float>(), "tau_gaba")
			("tau_nmda", po::value<float>(), "tau_nmda")
			("ampa_nmda_ratio", po::value<float>(), "ampa_nmda_ratio")
			("tau_vtrace", po::value<float>(), "tau_vtrace")
			("tau_cexc", po::value<float>(), "tau_cexc")
			("tau_cinh", po::value<float>(), "tau_cinh")
			("wee", po::value<float>(), "initial ee weights")
			("wei", po::value<float>(), "initial ei weights")
			("wie", po::value<float>(), "initial ie weights")
			("wii", po::value<float>(), "initial ii weights")
			("rule", po::value< std::string >(), "plasticity rules for EE and IE, assuming common shared weights, to enter as a string with separator a, one a at the beginning")
			("nh1", po::value<int>(), "size of first hidden layer in MLP")
			("nh2", po::value<int>(), "size of second hidden layer in MLP")
			("wmax", po::value<float>(), "max exc weight")
			("sparseness", po::value<float>(), "sparseness of all 4 recurrent connection types")
			("lns", po::value<float>(), "length_no_scoring")
			("ls", po::value<float>(), "length_scoring")
			("N_inputs", po::value<int>(), "number of input neurons")
			("sparseness_poisson", po::value<float>(), "sparseness of incoming poisson inputs to exc and inh  neurons")
			("w_poisson", po::value<float>(), "weights from inputs to exc and inh neurons")
			("poisson_rate", po::value<float>(), "poisson_rate in Hz")
			("min_rate_checker", po::value<float>(), "min_rate_checker in Hz")
			("max_rate_checker", po::value<float>(), "max_rate_checker in Hz")
			("tau_checker", po::value<float>(), "tau_checker in s")
			("workdir", po::value<std::string>(), "workdir to write output files (until we have a writeless monitor)")
			("n_recorded", po::value<int>(), "how many exc neurons to record")
			("record_i", po::value<bool>(), " whetyher to record inhibitory spikes or not")
			("n_recorded_i", po::value<int>(), "how many inh neurons to record, relevant only if record_i is true")
        ;

        po::variables_map vm;        
        po::store(po::parse_command_line(ac, av, desc), vm);

		if (vm.count("ID")) {ID= vm["ID"].as<std::string>();}
		if (vm.count("NE")) {NE = vm["NE"].as<int>();}
		if (vm.count("NI")) {NI = vm["NI"].as<int>();}
		if (vm.count("tau_ampa")) {tau_ampa = vm["tau_ampa"].as<float>();}
		if (vm.count("tau_gaba")) {tau_gaba = vm["tau_gaba"].as<float>();}
		if (vm.count("tau_nmda")) {tau_nmda = vm["tau_nmda"].as<float>();}
		if (vm.count("ampa_nmda_ratio")) {ampa_nmda_ratio = vm["ampa_nmda_ratio"].as<float>();}
		if (vm.count("tau_vtrace")) {tau_vtrace = vm["tau_vtrace"].as<float>();}
		if (vm.count("tau_cexc")) {tau_cexc = vm["tau_cexc"].as<float>();}
		if (vm.count("tau_cinh")) {tau_cinh = vm["tau_cinh"].as<float>();}
		if (vm.count("wee")) {wee = vm["wee"].as<float>();}
		if (vm.count("wei")) {wei = vm["wei"].as<float>();}
		if (vm.count("wie")) {wie = vm["wie"].as<float>();}
		if (vm.count("wii")) {wii = vm["wii"].as<float>();}
		if (vm.count("rule")) {rule_str = vm["rule"].as< std::string >();}
		if (vm.count("nh1")) {nh1 = vm["nh1"].as<int>();}
		if (vm.count("nh2")) {nh2 = vm["nh2"].as<int>();}
		if (vm.count("wmax")) {wmax = vm["wmax"].as<float>();}
		if (vm.count("sparseness")) {sparseness = vm["sparseness"].as<float>();}
		if (vm.count("lns")) {length_no_scoring = vm["lns"].as<float>();}
		if (vm.count("ls")) {length_scoring = vm["ls"].as<float>();}
		if (vm.count("N_inputs")) {N_inputs = vm["N_inputs"].as<int>();}
		if (vm.count("sparseness_poisson")) {sparseness_poisson = vm["sparseness_poisson"].as<float>();}
		if (vm.count("w_poisson")) {w_poisson = vm["w_poisson"].as<float>();}
		if (vm.count("poisson_rate")) {poisson_rate = vm["poisson_rate"].as<float>();}
		if (vm.count("min_rate_checker")) {min_rate_checker = vm["min_rate_checker"].as<float>();}
		if (vm.count("max_rate_checker")) {max_rate_checker = vm["max_rate_checker"].as<float>();}
		if (vm.count("tau_checker")) {tau_checker = vm["tau_checker"].as<float>();}
		if (vm.count("workdir")) {workdir = vm["workdir"].as<std::string>();}
		if (vm.count("n_recorded")) {n_recorded = vm["n_recorded"].as<int>();}
		if (vm.count("record_i")) {record_i= vm["record_i"].as<bool>();}
		if (vm.count("n_recorded_i")) {n_recorded_i= vm["n_recorded_i"].as<int>();}
	}
	catch(std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    catch(...) {
        std::cerr << "Exception of unknown type!\n";
    }

	//eta, 4 timescales, first layer, second layer, and last layer with bias (twice, once for on pre update, once for on post)
	//but 2 rules, they dont share eta, timescales, and last layer
	int expected_rule_size_agg = 2*5 + nh1*6 + nh2*nh1 + 2*2*(nh2+1); 
	int expected_rule_size = 5 + nh1*6 + nh2*nh1 + 2*(nh2+1); 

	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	// std::cout << "inside auryn, putative size of rule " << expected_rule_size << std::endl; 
	// std::cout << "inside auryn, putative size of rule agg " << expected_rule_size_agg << std::endl; 
	// printf("%s\n", rule_str.c_str());
	//////////////////////////////////////////////////////////////////////////////////////////////////////////

	rule_agg.resize(expected_rule_size_agg); 
	rule_agg = parse_input_plasticity(rule_str, expected_rule_size_agg);

	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	// std::cout << " rule_agg " << std::endl;
	// for (int i=0; i<expected_rule_size_agg; i++){
	// 	std::cout << rule_agg[i] << ", ";
	// }
	// std::cout << " END " << std::endl;
	//////////////////////////////////////////////////////////////////////////////////////////////////////////

	rule_EE.resize(expected_rule_size); 
	rule_IE.resize(expected_rule_size);

	//etas
	rule_EE[0] = rule_agg[0];
	rule_IE[0] = rule_agg[1];

	//taus EE
	rule_EE[1] = rule_agg[2];
	rule_EE[2] = rule_agg[3];
	rule_EE[3] = rule_agg[4];
	rule_EE[4] = rule_agg[5];

	//taus IE
	rule_IE[1] = rule_agg[6];
	rule_IE[2] = rule_agg[7];
	rule_IE[3] = rule_agg[8];
	rule_IE[4] = rule_agg[9];

	//shared weights
	for (int i = 0; i<nh1*6 + nh2*nh1; i++){
		rule_EE[i+5] = rule_agg[i+10];
		rule_IE[i+5] = rule_agg[i+10];
	}

	// //last layers
	for (int i=0; i < 2*(nh2+1); i++){
		rule_EE[i + 5 + nh1*6 + nh2*nh1] = rule_agg[i + 2*5 + nh1*6 + nh2*nh1];
		rule_IE[i + 5 + nh1*6 + nh2*nh1] = rule_agg[i + 2*5 + nh1*6 + nh2*nh1 + 2*(nh2+1)];
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	// std::cout << "inside auryn, putative size of rule agg " << expected_rule_size_agg << std::endl; 
	// std::cout << "putative size of rule " << expected_rule_size << std::endl; 
	// std::cout << " rule_EE " << std::endl;
	// for (int i=0; i<expected_rule_size; i++){
	// 	std::cout << rule_EE[i] << ", ";
	// }
	// std::cout << " END " << std::endl;
	// std::cout << " rule_IE " << std::endl;
	// for (int i=0; i<expected_rule_size; i++){
	// 	std::cout << rule_IE[i] << ", ";
	// }
	// std::cout << " END " << std::endl;
	//////////////////////////////////////////////////////////////////////////////////////////////////////////


	///////////////////////
	// Build the network //
	///////////////////////

	auryn_init(ac, av, workdir.c_str(), "default", "", NONE, NONE);
	sys->quiet = true;

	CVAIFGroup* neurons_e = new CVAIFGroup(NE);
	neurons_e->set_tau_ampa(tau_ampa);
	neurons_e->set_tau_gaba(tau_gaba);
	neurons_e->set_tau_nmda(tau_nmda);
	neurons_e->set_ampa_nmda_ratio(ampa_nmda_ratio);
	neurons_e->set_tau_vtrace(tau_cexc);
	neurons_e->set_tau_cexc(tau_cexc);
	neurons_e->set_tau_cinh(tau_cinh);

	CVAIFGroup* neurons_i = new CVAIFGroup(NI);
	neurons_i->set_tau_ampa(tau_ampa);
	neurons_i->set_tau_gaba(tau_gaba);
	neurons_i->set_tau_nmda(tau_nmda);
	neurons_i->set_ampa_nmda_ratio(ampa_nmda_ratio);
	neurons_i->set_tau_vtrace(tau_cexc);
	neurons_i->set_tau_cexc(tau_cexc);
	neurons_i->set_tau_cinh(tau_cinh);

	// external inputs
	PoissonGroup * poisson = new PoissonGroup(N_inputs, poisson_rate);
	SparseConnection * con_ext_exc = new SparseConnection(poisson, neurons_e, w_poisson, sparseness_poisson, GLUT);
	SparseConnection * con_ext_inh = new SparseConnection(poisson, neurons_i, w_poisson, sparseness_poisson, GLUT);

	// recurrent connectivity
	T4wvceciMLPConnection* con_ee = new T4wvceciMLPConnection(neurons_e, neurons_e, wee, sparseness, rule_EE, nh1, nh2, wmax, GLUT, "T4wvceciMLPConnectionEE");
	SparseConnection * con_ei = new SparseConnection(neurons_e, neurons_i, wei, sparseness, GLUT);
	T4wvceciMLPConnection* con_ie = new T4wvceciMLPConnection(neurons_i, neurons_e, wie, sparseness, rule_IE, nh1, nh2, wmax, GABA, "T4wvceciMLPConnectionIE");
	SparseConnection * con_ii = new SparseConnection(neurons_i, neurons_i, wii, sparseness, GABA);

	// rate checker and loss calculation when during scoring phase
	RateChecker* cs = new RateChecker(neurons_e, 0, max_rate_checker, tau_checker);

	///////////////////////////////////////////////////////
	// Run the network for the training time, no scoring //
	///////////////////////////////////////////////////////
	
	sys->run(length_no_scoring);
	
	//////////////////////////////////////////
	// Run the network for the scoring time //
	//////////////////////////////////////////

	SpikeMonitor* smon_e = new SpikeMonitor(neurons_e , sys->fn("out.e." + ID, "ras"), n_recorded);
	if (record_i == true){
		SpikeMonitor* smon_i = new SpikeMonitor(neurons_i , sys->fn("out.i." + ID, "ras"), n_recorded_i);
	}
	WeightMonitor* wmon_ee = new WeightMonitor(con_ee, sys->fn("con_ee." + ID,"syn"), 0.1);
	wmon_ee->add_equally_spaced(100);
	WeightMonitor* wmon_ie = new WeightMonitor(con_ie, sys->fn("con_ie." + ID,"syn"), 0.1);
	wmon_ie->add_equally_spaced(100);

	sys->run(length_scoring);

	///////////////// DEBUG TO REMOVE IN BIG SIM //////////////////////////////////////////////////////////////////////////////////// CAREFUL
	// con_ee->write_to_file(sys->fn("full_ee","syn"));
	// con_ie->write_to_file(sys->fn("full_ie","syn"));
	///////////////// DEBUG TO REMOVE IN BIG SIM //////////////////////////////////////////////////////////////////////////////////// CAREFUL

	/////////////////////////////////
	// Compute and return the loss //
	/////////////////////////////////

	if (sys->get_time() >= (length_scoring + length_no_scoring)){
		std::cout << "cynthia" << -1 << "cynthia";
	}
	else{
		std::cout << "cynthia" << sys->get_time() << "cynthia";
	}
    
	auryn_free();
	return 0;
}