#include "auryn.h"
#include "SixParamConnection.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>

namespace po = boost::program_options;
using namespace auryn;


std::vector<float> parse_input_plasticity(std::vector<std::string> rule_str)
{
	// parsing the command line rule argument: (it is given as a string because multitoken() is bugged: negative numbers cause errors)
	std::vector<float> rule(6);
	std::string s = rule_str[0];
	std::string delimiter = "a";
	size_t pos = 0;
	int ct = 0;
	std::string token;

	token = s.substr(0, pos); // remove the first a (needed in case first param is negative, or is it?)
	s.erase(0, pos + delimiter.length());

	while ((pos = s.find(delimiter)) != std::string::npos) { // parse the rest of the expression
		token = s.substr(0, pos);
		rule[ct] = boost::lexical_cast<float>(token);
		s.erase(0, pos + delimiter.length());
		ct ++;
	}
	return(rule);
}


int main(int ac, char* av[])
{
	/////////////////////////////////////////////////////////
	// Get simulation parameters from command line options //
	/////////////////////////////////////////////////////////

	std::string ID = "0";
	std::string workdir;

	float wee = 0.;
	float wei = 0.;
	float wie = 0.;
	float wii = 0.;
	float sparseness = 0.;

	float sparseness_poisson = 0.;
	float rate_poisson = 0.;
	float weight_poisson = 0.;

	std::vector<float> ruleEE(6);
	std::vector<std::string> ruleEE_str;
	std::vector<float> ruleEI(6);
	std::vector<std::string> ruleEI_str;
	std::vector<float> ruleIE(6);
	std::vector<std::string> ruleIE_str;
	std::vector<float> ruleII(6);
	std::vector<std::string> ruleII_str;
	float eta = 0.;
	float wmax = 0.;

	float length_no_scoring = 0.;
	float length_scoring = 0.;

	int NE = 0;
	int NI = 0;
	int N_input = 0;
	float tau_ampa = 0;
	float tau_gaba = 0;
	float tau_nmda = 0;
	float ampa_nmda_ratio = 0;

	float max_rate_checker = 0.;
	float tau_checker = 0.;

	int n_recorded = 100;
	bool record_i = false;
	int n_recorded_i = 500;

    try {
        po::options_description desc("Allowed options");
        desc.add_options()
			("ID", po::value<std:: string>(), "ID to name the monitor output files correctly")
			("NE", po::value<int>(), "NE")
			("NI", po::value<int>(), "NI")
			("tau_ampa", po::value<float>(), "tau_ampa")
			("tau_gaba", po::value<float>(), "tau_gaba")
			("tau_nmda", po::value<float>(), "tau_nmda")
			("ampa_nmda_ratio", po::value<float>(), "ampa_nmda_ratio")
			("ruleEE", po::value< std::vector<std::string> >(), "plasticity rule for EE, to enter as a string with separator a (start with a)")
			("ruleEI", po::value< std::vector<std::string> >(), "plasticity rule for EI")
			("ruleIE", po::value< std::vector<std::string> >(), "plasticity rule for IE")
			("ruleII", po::value< std::vector<std::string> >(), "plasticity rule for II")
			("eta", po::value<float>(), "lerning rate for the rule")
			("wmax", po::value<float>(), "max exc weight")
			("wee", po::value<float>(), "wee")
			("wei", po::value<float>(), "wei")
			("wie", po::value<float>(), "wie")
			("wii", po::value<float>(), "wii")
			("sparseness", po::value<float>(), "sparseness")
			("N_input", po::value<int>(), "N_input")
			("sparseness_poisson", po::value<float>(), "sparseness_poisson")
			("rate_poisson", po::value<float>(), "rate_poisson")
			("weight_poisson", po::value<float>(), "weight_poisson")
			("max_rate_checker", po::value<float>(), "max_rate_checker")
			("tau_checker", po::value<float>(), "tau_checker")
			("lns", po::value<float>(), "length_no_scoring")
			("ls", po::value<float>(), "length_scoring")
			("workdir", po::value<std::string>(), "workdir to write output files (until we have a writeless monitor)")
			("n_recorded", po::value<int>(), "how many exc neurons to record")
			("record_i", po::value<bool>(), " whetyher to record inhibitory spikes or not")
			("n_recorded_i", po::value<int>(), "how many inh neurons to record, relevant only if record_i is true")
        ;

        po::variables_map vm;
        po::store(po::parse_command_line(ac, av, desc), vm);

		if (vm.count("ID")) {ID= vm["ID"].as<std::string>();}
		if (vm.count("NE")) {NE= vm["NE"].as<int>();}
		if (vm.count("NI")) {NI= vm["NI"].as<int>();}
		if (vm.count("tau_ampa")) {tau_ampa = vm["tau_ampa"].as<float>();}
		if (vm.count("tau_gaba")) {tau_gaba = vm["tau_gaba"].as<float>();}
		if (vm.count("tau_nmda")) {tau_nmda = vm["tau_nmda"].as<float>();}
		if (vm.count("ampa_nmda_ratio")) {ampa_nmda_ratio = vm["ampa_nmda_ratio"].as<float>();}
		if (vm.count("ruleEE")) {ruleEE_str = vm["ruleEE"].as< std::vector<std::string> >();}
		if (vm.count("ruleEI")) {ruleEI_str = vm["ruleEI"].as< std::vector<std::string> >();}
		if (vm.count("ruleIE")) {ruleIE_str = vm["ruleIE"].as< std::vector<std::string> >();}
		if (vm.count("ruleII")) {ruleII_str = vm["ruleII"].as< std::vector<std::string> >();}
		if (vm.count("eta")) {eta = vm["eta"].as<float>();}
		if (vm.count("wmax")) {wmax = vm["wmax"].as<float>();}
		if (vm.count("wee")) {wee = vm["wee"].as<float>();}
		if (vm.count("wei")) {wei = vm["wei"].as<float>();}
		if (vm.count("wie")) {wie = vm["wie"].as<float>();}
		if (vm.count("wii")) {wii = vm["wii"].as<float>();}
		if (vm.count("sparseness")) {sparseness = vm["sparseness"].as<float>();}
		if (vm.count("N_input")) {N_input= vm["N_input"].as<int>();}
		if (vm.count("sparseness_poisson")) {sparseness_poisson = vm["sparseness_poisson"].as<float>();}
		if (vm.count("rate_poisson")) {rate_poisson = vm["rate_poisson"].as<float>();}
		if (vm.count("weight_poisson")) {weight_poisson = vm["weight_poisson"].as<float>();}
		if (vm.count("max_rate_checker")) {max_rate_checker = vm["max_rate_checker"].as<float>();}
		if (vm.count("tau_checker")) {tau_checker = vm["tau_checker"].as<float>();}
		if (vm.count("lns")) {length_no_scoring = vm["lns"].as<float>();}
		if (vm.count("ls")) {length_scoring = vm["ls"].as<float>();}
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

	// parsing the command line rule arguments
	ruleEE = parse_input_plasticity(ruleEE_str);
	ruleEI = parse_input_plasticity(ruleEI_str);
	ruleIE = parse_input_plasticity(ruleIE_str);
	ruleII = parse_input_plasticity(ruleII_str);

	///////////////////////
	// Build the network //
	///////////////////////

	auryn_init(ac, av, workdir.c_str(), "default", "", NONE, NONE);
	sys->quiet = true;

	// handle randomness of simulation: by default random seed
	std::srand(std::time(0));
	sys->set_master_seed(std::rand());

	IFGroup* neurons_e = new IFGroup(NE);
	neurons_e->set_tau_ampa(tau_ampa); //5e-3
	neurons_e->set_tau_gaba(tau_gaba); //10e-3
	neurons_e->set_tau_nmda(tau_nmda); //100e-3
	neurons_e->set_ampa_nmda_ratio(ampa_nmda_ratio); //0.3

	IFGroup* neurons_i = new IFGroup(NI);
	neurons_i->set_tau_ampa(tau_ampa); //5e-3
	neurons_i->set_tau_gaba(tau_gaba); //10e-3
	neurons_i->set_tau_nmda(tau_nmda); //100e-3
	neurons_i->set_ampa_nmda_ratio(ampa_nmda_ratio); //0.3

	// Checker scorer to stop simulations that exceed certain firing rates
	RateChecker* cs = new RateChecker(neurons_e, 0, max_rate_checker, tau_checker);

	// external inputs to the neurons
	PoissonGroup* poisson = new PoissonGroup(N_input, rate_poisson);
	SparseConnection* con_ext_exc = new SparseConnection(poisson,neurons_e,weight_poisson,sparseness_poisson,GLUT);
	SparseConnection* con_ext_inh = new SparseConnection(poisson,neurons_i, weight_poisson, sparseness_poisson, GLUT);

	// recurrent connectivity
	SixParamConnection* con_ee = new SixParamConnection(neurons_e,
														neurons_e,
														wee,
														sparseness,
														eta,
														ruleEE[2],
														ruleEE[3],
														ruleEE[4],
														ruleEE[5],
														ruleEE[0],
														ruleEE[1],
														wmax,
														GLUT);
	SixParamConnection* con_ei = new SixParamConnection(neurons_e,
														neurons_i,
														wei,
														sparseness,
														eta,
														ruleEI[2],
														ruleEI[3],
														ruleEI[4],
														ruleEI[5],
														ruleEI[0],
														ruleEI[1],
														wmax,
														GLUT);
	SixParamConnection* con_ie = new SixParamConnection(neurons_i,
														neurons_e,
														wie,
														sparseness,
														eta,
														ruleIE[2],
														ruleIE[3],
														ruleIE[4],
														ruleIE[5],
														ruleIE[0],
														ruleIE[1],
														wmax,
														GABA);
	SixParamConnection* con_ii = new SixParamConnection(neurons_i,
														neurons_i,
														wii,
														sparseness,
														eta,
														ruleII[2],
														ruleII[3],
														ruleII[4],
														ruleII[5],
														ruleII[0],
														ruleII[1],
														wmax,
														GABA);
	
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
	WeightMonitor* wmon_ei = new WeightMonitor(con_ei, sys->fn("con_ei." + ID,"syn"), 0.1);
	wmon_ei->add_equally_spaced(100);
	WeightMonitor* wmon_ie = new WeightMonitor(con_ie, sys->fn("con_ie." + ID,"syn"), 0.1);
	wmon_ie->add_equally_spaced(100);
	WeightMonitor* wmon_ii = new WeightMonitor(con_ii, sys->fn("con_ii." + ID,"syn"), 0.1);
	wmon_ii->add_equally_spaced(100);

	sys->run(length_scoring);

	///////////////// DEBUG TO REMOVE IN BIG SIM //////////////////////////////////////////////////////////////////////////////////// CAREFUL
	// con_ee->write_to_file(sys->fn("full_ee","syn"));
	// con_ei->write_to_file(sys->fn("full_ei","syn"));
	// con_ie->write_to_file(sys->fn("full_ie","syn"));
	// con_ii->write_to_file(sys->fn("full_ii","syn"));
	///////////////// DEBUG TO REMOVE IN BIG SIM //////////////////////////////////////////////////////////////////////////////////// CAREFUL


	if (sys->get_time() >= (length_scoring + length_no_scoring)){
		std::cout << "cynthia" << -1 << "cynthia";
	}
	else{
		std::cout << "cynthia" << sys->get_time() << "cynthia";
	}

	

	auryn_free();
	return 0;
}