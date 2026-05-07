#include <boost/program_options.hpp>
#include <iostream>

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

    std::size_t max_events = 1000;

    int ncore = 13;
	bool do_roots_n2 = true;
	bool do_roots_n3 = false;
	
	bool do_cumulants_n2 = true;
	bool do_cumulants_n3 = false;
	
};

void LYZ(const char* input_filename,
         const char* output_filename,
         const LYZParameters& par);

int main(int argc, char* argv[])
{
    try {

        std::string input_file;
        std::string output_file;

        LYZParameters par;

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
             "Maximum number of events")

            ("ncore",
             po::value<int>(&par.ncore)
                 ->default_value(par.ncore),
             "Number of OpenMP threads")

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
				 "Calculate and save n=3 cumulants");

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
