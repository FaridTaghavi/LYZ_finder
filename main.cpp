#include <boost/program_options.hpp>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace po = boost::program_options;

struct LYZParameters {

    double re_min = 30.0;
    double re_max = 80.0;
    double dre    = 15.0;

    double im_min = 0.0;
    double im_max = 30.0;
    double dim    = 10.0;

    int Nsub = 10;
    int Nres = 500;

    std::size_t max_events = 0;

    int ncore = 13;
    unsigned int seed = 12345;
    std::string root_algorithm = "hybridS";
    std::vector<std::pair<double, double>> root_start_points;
    bool multicore_root_search = false;
	bool do_roots_n2 = true;
	bool do_roots_n3 = false;
	
	bool do_cumulants_n2 = true;
	bool do_cumulants_n3 = false;

    bool do_theta_n2 = false;
    bool do_theta_n3 = false;
    double theta_k_max = 200.0;
    double theta_dk = 0.25;
    int theta_degree = 4;
    int epsilon_bins = 0;
	
};

void LYZ(const char* input_filename,
         const char* output_filename,
         const LYZParameters& par);

std::pair<double, double> ParseRootStartPoint(const std::string& text)
{
    std::string normalized = text;
    for (char& c : normalized) {
        if (c == ':' || c == ';') {
            c = ',';
        }
    }

    std::stringstream ss(normalized);
    std::string re_text;
    std::string im_text;

    if (!std::getline(ss, re_text, ',') ||
        !std::getline(ss, im_text, ',') ||
        re_text.empty() ||
        im_text.empty()) {
        throw std::runtime_error(
            "Invalid --root-start-point '" + text + "'. Use Re,Im, for example 40,5."
        );
    }

    std::string extra;
    if (std::getline(ss, extra, ',')) {
        throw std::runtime_error(
            "Invalid --root-start-point '" + text + "'. Use exactly two values."
        );
    }

    return {std::stod(re_text), std::stod(im_text)};
}

int main(int argc, char* argv[])
{
    try {

        std::string input_file;
        std::string output_file;

        LYZParameters par;
        std::vector<std::string> root_start_point_args;

        po::options_description desc("LYZ options");

        desc.add_options()

            ("help,h", "Print help message")

            ("input,i",
             po::value<std::string>(&input_file)
                 ->default_value("PbPb_events.root"),
             "Input ROOT file")

            ("output,o",
             po::value<std::string>(&output_file)
                 ->default_value("bootstrap_roots.dat"),
             "Output data file")

            ("re-min",
             po::value<double>(&par.re_min)
                 ->default_value(par.re_min),
             "Minimum Re(k)")

            ("re-max",
             po::value<double>(&par.re_max)
                 ->default_value(par.re_max),
             "Maximum Re(k)")

            ("dre",
             po::value<double>(&par.dre)
                 ->default_value(par.dre),
             "Step size in Re(k)")

            ("im-min",
             po::value<double>(&par.im_min)
                 ->default_value(par.im_min),
             "Minimum Im(k)")

            ("im-max",
             po::value<double>(&par.im_max)
                 ->default_value(par.im_max),
             "Maximum Im(k)")

            ("dim",
             po::value<double>(&par.dim)
                 ->default_value(par.dim),
             "Step size in Im(k)")

            ("nsub",
             po::value<int>(&par.Nsub)
                 ->default_value(par.Nsub),
             "Bootstrap subsamples")

            ("nres",
             po::value<int>(&par.Nres)
                 ->default_value(par.Nres),
             "Bootstrap resampling count")

            ("max-events",
             po::value<std::size_t>(&par.max_events)
                 ->default_value(par.max_events),
             "Maximum number of events; 0 means use all available events")

            ("ncore",
             po::value<int>(&par.ncore)
                 ->default_value(par.ncore),
             "Number of OpenMP threads")

            ("multicore-root-search",
             po::value<bool>(&par.multicore_root_search)
                 ->default_value(par.multicore_root_search),
             "Use OpenMP threads over Re/Im root-start points instead of bootstrap resampling")

            ("seed",
             po::value<unsigned int>(&par.seed)
                 ->default_value(par.seed),
             "Base RNG seed; bootstrap sample i uses seed + i")

            ("root-algorithm",
             po::value<std::string>(&par.root_algorithm)
                 ->default_value(par.root_algorithm),
             "Root finder algorithm: hybridS, hybrid, or hybridSJ")

            ("root-start-point",
             po::value<std::vector<std::string>>(&root_start_point_args)
                 ->composing(),
             "Root finder initial guess Re,Im. Can be repeated, e.g. --root-start-point 40,0 --root-start-point 55,10. If omitted, the Re/Im range grid is used.")
				
					("do-roots-n2",
				 po::value<bool>(&par.do_roots_n2)
				     ->default_value(par.do_roots_n2),
				 "Calculate and save n=2 roots")
				
			("do-roots-n3",
				 po::value<bool>(&par.do_roots_n3)
				     ->default_value(par.do_roots_n3),
				 "Calculate and save n=3 roots")
				
			("do-cumulants-n2",
				 po::value<bool>(&par.do_cumulants_n2)
				     ->default_value(par.do_cumulants_n2),
				 "Calculate and save n=2 cumulants")
				
			("do-cumulants-n3",
				 po::value<bool>(&par.do_cumulants_n3)
				     ->default_value(par.do_cumulants_n3),
				 "Calculate and save n=3 cumulants")

            ("do-theta-n2",
                 po::value<bool>(&par.do_theta_n2)
                     ->default_value(par.do_theta_n2),
                 "Scan G_2(k), find log|G_2(k)| peaks, and fit theta_2(k)")

            ("do-theta-n3",
                 po::value<bool>(&par.do_theta_n3)
                     ->default_value(par.do_theta_n3),
                 "Scan G_3(k), find log|G_3(k)| peaks, and fit theta_3(k)")

            ("theta-k-max",
                 po::value<double>(&par.theta_k_max)
                     ->default_value(par.theta_k_max),
                 "Maximum real k used for the theta(k) scan")

            ("theta-dk",
                 po::value<double>(&par.theta_dk)
                     ->default_value(par.theta_dk),
                 "Real-k step size used for the theta(k) scan")

            ("theta-degree",
                 po::value<int>(&par.theta_degree)
                     ->default_value(par.theta_degree),
                 "Polynomial degree used to fit theta(k) from log|G(k)| peaks")

            ("epsilon-bins",
                 po::value<int>(&par.epsilon_bins)
                     ->default_value(par.epsilon_bins),
                 "Use this many epsilon bins for root finding; 0 uses all events exactly");

        po::variables_map vm;

        po::store(
            po::parse_command_line(argc, argv, desc),
            vm
        );

        po::notify(vm);

		        if (vm.count("help")) {

	            std::cout << desc << "\n";

		            return 0;
		        }

		        if (par.root_algorithm != "hybridS" &&
		            par.root_algorithm != "hybrid" &&
		            par.root_algorithm != "hybridSJ") {

		            throw std::runtime_error(
		                "Invalid --root-algorithm. Use hybridS, hybrid, or hybridSJ."
		            );
		        }

                for (const auto& arg : root_start_point_args) {
                    par.root_start_points.push_back(ParseRootStartPoint(arg));
                }

                if (par.theta_k_max <= 0.0) {
                    throw std::runtime_error("--theta-k-max must be positive.");
                }

                if (par.theta_dk <= 0.0) {
                    throw std::runtime_error("--theta-dk must be positive.");
                }

                if (par.theta_degree < 0) {
                    throw std::runtime_error("--theta-degree must be non-negative.");
                }

                if (par.epsilon_bins < 0) {
                    throw std::runtime_error("--epsilon-bins must be non-negative.");
                }

		        LYZ(
            input_file.c_str(),
            output_file.c_str(),
            par
        );
    }

    catch (const std::exception& e) {

        std::cerr << "Error: "
                  << e.what()
                  << "\n";

        return 1;
    }

    return 0;
}
