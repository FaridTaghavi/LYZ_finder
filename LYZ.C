#include <cmath>
#include <complex>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>
#include <random>


#include "TCanvas.h"
#include "TGraph.h"
#include "TMath.h"
#include "TAxis.h"
#include <TFile.h>
#include <TTree.h>
#include <TTreeReader.h>
#include <TTreeReaderValue.h>
#include <TTreeReaderArray.h>
#include <TH2D.h>
#include <TError.h>
#include <gsl/gsl_errno.h>
#include "Math/Functor.h"
#include "Math/RootFinder.h"

#include "complex_bessel.h"

 
#include "RConfigure.h"
 
#ifdef R__HAS_MATHMORE
#include "Math/MultiRootFinder.h"
#else
#error libMathMore is not available - cannot run this tutorial
#endif
#include "Math/WrappedMultiTF1.h"
#include "TF2.h"
#include "TError.h"
#include <Math/GSLMultiRootFinder.h> 
#include <Math/IFunction.h>
#include <chrono>

#include <omp.h>
#include <limits>
#include <exception>
#include <algorithm>

#include "event_CLASS.h"
#include <filesystem>
namespace {

std::complex<double> J0_complex(const std::complex<double>& z)
{
    BesselErrors error;
    std::complex<double> value = sp_bessel::besselJ(0.0, z, false, &error);

    if (error.errorCode != Success) {
        throw std::runtime_error("complex_bessel::besselJ failed: " + error.errorMessage);
    }

    return value;
}

std::complex<double> J1_complex(const std::complex<double>& z)
{
    BesselErrors error;
    std::complex<double> value = sp_bessel::besselJ(1.0, z, false, &error);

    if (error.errorCode != Success) {
        throw std::runtime_error("complex_bessel::besselJ failed: " + error.errorMessage);
    }

    return value;
}
} // namespace


void MeanJ0(double k, const std::vector<double>& e2_vals, double& result)
{
    double sum = 0.0;
    const std::size_t N = e2_vals.size();

    for (std::size_t i = 0; i < N; ++i) {
        sum += TMath::BesselJ0(k * e2_vals[i]);
    }

    result = (N > 0) ? sum / static_cast<double>(N) : 0.0;
}

void MeanJ0_complex(std::complex<double> k,
                    const std::vector<double>& e2_vals,
                    std::complex<double>& result)
{
    std::complex<double> sum = 0.0;
    const std::size_t N = e2_vals.size();

    for (std::size_t i = 0; i < N; ++i) {
        sum += J0_complex(k * e2_vals[i]);
    }

    result = (N > 0) ? sum / static_cast<double>(N)
                     : std::complex<double>(0.0, 0.0);
}



struct ReMeanJ0 {
    const std::vector<double>& e2_vals;
    
	ReMeanJ0(const std::vector<double>& vals)
        : e2_vals(vals) {}
    
	double operator()(const double* x) const {
        std::complex<double> k{x[0], x[1]};
		std::complex<double> result; 
        MeanJ0_complex(k, e2_vals, result);
        return result.real(); 
    }
};

struct ImMeanJ0 {
    const std::vector<double>& e2_vals;

	ImMeanJ0(const std::vector<double>& vals)
        : e2_vals(vals) {}
    
	double operator()(const double* x) const {
        std::complex<double> k{x[0], x[1]};
		std::complex<double> result; 
        MeanJ0_complex(k, e2_vals, result);
        return result.imag(); 
    }
};

struct RootResult {
    bool ok;
    int status;
    std::complex<double> k;
    std::complex<double> F;
    double time_ms;
};


RootResult FindComplexZero_noDerivative(
    const std::vector<double>& e2_list,
    double x0_re,
    double x0_im
) {
    RootResult result;
    result.ok = false;
    result.status = -999;
    result.k = {NAN, NAN};
    result.F = {NAN, NAN};
    result.time_ms = 0.0;

    try {
        ROOT::Math::GSLMultiRootFinder finder(
            ROOT::Math::GSLMultiRootFinder::kHybridS
        );

        ReMeanJ0 f_re(e2_list);
        ImMeanJ0 f_im(e2_list);

        finder.AddFunction(f_re, 2);
        finder.AddFunction(f_im, 2);

        double x0[2] = {x0_re, x0_im};

        auto t1 = std::chrono::high_resolution_clock::now();
        bool ok = finder.Solve(x0, 1000, 1e-10, 1e-10);
        auto t2 = std::chrono::high_resolution_clock::now();

        result.time_ms =
            std::chrono::duration<double, std::milli>(t2 - t1).count();

        const double* root = finder.X();
        const double* fval = finder.FVal();

        result.ok = ok;
        result.status = finder.Status();

        if (root && fval) {
            result.k = {root[0], root[1]};
            result.F = {fval[0], fval[1]};
        }
    }
    catch (const std::exception& e) {
        result.ok = false;
        result.status = -999;
    }

    return result;
}

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

struct VnPowers {
    double vn2to2;
    double vn4to4;
    double vn6to6;
    double vn8to8;
    double vn10to10;
};

struct averageV {
	double n2;
	double n4;
	double n6;
	double n8;
	double n10;
};

averageV ComputeMoments(const std::vector<double>& e)
{
    averageV av{0.0, 0.0, 0.0, 0.0, 0.0};

    const double N = static_cast<double>(e.size());
    if (N == 0.0) return av;

    for (double x : e) {
        double x2 = x * x;
        double x4 = x2 * x2;
        double x6 = x4 * x2;
        double x8 = x4 * x4;
        double x10 = x8 * x2;

        av.n2  += x2;
        av.n4  += x4;
        av.n6  += x6;
        av.n8  += x8;
        av.n10 += x10;
    }

    av.n2  /= N;
    av.n4  /= N;
    av.n6  /= N;
    av.n8  /= N;
    av.n10 /= N;

    return av;
}

VnPowers ComputeVnPowers(averageV av ) {
    
	VnPowers out;

    out.vn2to2 = av.n2;

    out.vn4to4 = 2.0*std::pow(av.n2, 2.) - av.n4;

    out.vn6to6 = (12.0*std::pow(av.n2, 3.) - 9. * av.n2*av.n4 + av.n6) /4.;

    out.vn8to8 = (18.0*(8.0*std::pow(av.n2, 4.0) - 8.0*std::pow(av.n2, 2.0)*av.n4 + std::pow(av.n4,2.0)) + 16.0*av.n2*av.n6 - av.n8)/33.;
    
	out.vn10to10 = (av.n10 + 5.0*(576.0*std::pow(av.n2,5.0) - 720.0 * std::pow(av.n2,3.0)* av.n4 + 80.0*std::pow(av.n2,2.0)*av.n6 - 20.0 * av.n4*av.n6 + 5.0*av.n2*(36.0*std::pow(av.n4,2.0) - av.n8)))/456.;

    return out;
}

bool IsGoodRoot(RootResult& r, double max_found_root_size)
{
    if (!r.ok) return false;
    if (r.status != 0) return false;

    if (!std::isfinite(r.k.real()) ||
        !std::isfinite(r.k.imag())) return false;

    if (std::abs(r.F) > 1e-8) return false;

    const double axis_tol = 1e-8;

    if (r.k.real() < -axis_tol) return false;
    if (r.k.imag() < -axis_tol) return false;

    if (std::abs(r.k.real()) < axis_tol)
        r.k.real(0.0);

    if (std::abs(r.k.imag()) < axis_tol)
        r.k.imag(0.0);

    if (std::abs(r.k) > max_found_root_size)
        return false;

    return true;
}

void LYZ(const char* input_filename,
         const char* output_folder,
         const LYZParameters& par)
{
	// Searching the root in (ReZ, ImZ) plane
    const double re_min = par.re_min;
    const double re_max = par.re_max;
    const double dre    = par.dre;

    const double im_min = par.im_min;
    const double im_max = par.im_max;
    const double dim    = par.dim;

	// Bootstrap
    const int Nsub = par.Nsub;
    const int Nres = par.Nres;

	const bool do_roots_n2 = par.do_roots_n2;
	const bool do_roots_n3 = par.do_roots_n3;

	const bool do_cumulants_n2 = par.do_cumulants_n2;
	const bool do_cumulants_n3 = par.do_cumulants_n3;

    const std::size_t max_events = par.max_events;

	// Numebr of cores in searching for the roots
    const int ncore = par.ncore;	
	
	const double max_found_root_size = 2 * std::sqrt( re_max * re_max + im_max * im_max ); // sometimes the rootfinder finds very big roots, we discard them. 
	
	TFile file(input_filename, "READ");
	
	if (file.IsZombie()) {
	    std::cerr << "Error: cannot open ROOT file "
	              << input_filename << "\n";
	    return;
	}
    
	TTree* tree = nullptr;
	file.GetObject("trento_events", tree);
	
	event_CLASS* event = nullptr;
	tree->SetBranchAddress("events", &event);
	
	
	const std::size_t nevents =
	    std::min<std::size_t>(max_events, tree->GetEntries());

	
	// Keeping unbunched events are needed anymore!
	// std::vector<double> e2_list;
	// e2_list.reserve(nevents);
		

	std::vector<std::vector<double>> subsamples_e2(Nsub);
	std::vector<std::vector<double>> subsamples_e3(Nsub);


	// Reading ttree and fill the subsample	
	for (std::size_t i = 0; i < nevents; ++i) {
	   
	   	tree->GetEntry(i);
	
	    // double ex2 = event->Get_epsilonx(2);
	    // double ey2 = event->Get_epsilony(2);
	
	    // double e2 = std::sqrt(ex2 * ex2 + ey2 * ey2);
		
    	size_t isub = i % Nsub;
		
		if (do_roots_n2 || do_cumulants_n2) {
		    double ex2 = event->Get_epsilonx(2);
		    double ey2 = event->Get_epsilony(2);
		    double e2 = std::sqrt(ex2 * ex2 + ey2 * ey2);
    		subsamples_e2[isub].push_back(e2);
		}
		
		if (do_roots_n3 || do_cumulants_n3) {
		    double ex3 = event->Get_epsilonx(3);
		    double ey3 = event->Get_epsilony(3);
		    double e3 = std::sqrt(ex3 * ex3 + ey3 * ey3);
    		subsamples_e3[isub].push_back(e3);
		}

		// e2_list.push_back(e2); //Not needed anymore!

	}
	
	
	std::mt19937 rng(12345);
	std::uniform_int_distribution<int> dist(0, Nsub - 1);
	
	std::vector<std::vector<RootResult>> roots_n2_from_resampling;
	std::vector<std::vector<RootResult>> roots_n3_from_resampling;
	std::vector<VnPowers> cumulants_n2_from_resampling;
	std::vector<VnPowers> cumulants_n3_from_resampling;
	
	for (int ires = 0; ires < Nres; ++ires) {
	
	    std::vector<double> e2_bootstrap, e3_bootstrap;
	
		if (do_roots_n2 || do_cumulants_n2) {
	    	
			e2_bootstrap.reserve(nevents);
	    	for (int isub = 0; isub < Nsub; ++isub) {
	    	    int pick = dist(rng);
	
	    	    e2_bootstrap.insert(
	    	        e2_bootstrap.end(),
	    	        subsamples_e2[pick].begin(),
	    	        subsamples_e2[pick].end()
	    	    );
	    	}
		}
		if (do_roots_n3 || do_cumulants_n3) {
	    	
			e3_bootstrap.reserve(nevents);
	    	for (int isub = 0; isub < Nsub; ++isub) {
	    	    int pick = dist(rng);
	
	    	    e3_bootstrap.insert(
	    	        e3_bootstrap.end(),
	    	        subsamples_e3[pick].begin(),
	    	        subsamples_e3[pick].end()
	    	    );
	    	}
		}
		std::cout << "----------------------------------\n";	
    	std::cout << "Bootstrap sample " << ires + 1
    	          << " out of " << Nres
    	          << " resampling.\n";	
		std::cout << "----------------------------------\n";	



		// <<<<<<<<<<<<<<< Find Cumulants >>>>>>>>>>>>>>>>>>>
		
		
		if (do_cumulants_n2) {
		    averageV av2 = ComputeMoments(e2_bootstrap);
		    VnPowers c2 = ComputeVnPowers(av2);
		    cumulants_n2_from_resampling.push_back(c2);
		}
		
		if (do_cumulants_n3) {
		    averageV av3 = ComputeMoments(e3_bootstrap);
		    VnPowers c3 = ComputeVnPowers(av3);
		    cumulants_n3_from_resampling.push_back(c3);
		}


		// <<<<<<<<<<<<<<<  Find roots >>>>>>>>>>>>>>>

		std::vector<RootResult> roots_n2, roots_n3;
	
	
		int Nre = static_cast<int>((re_max - re_min) / dre);
		int Nim = static_cast<int>((im_max - im_min) / dim);

		// Turn off the root finder warning info
		gsl_set_error_handler_off();
		gErrorIgnoreLevel = kWarning;
		gErrorIgnoreLevel = kFatal;	
	
		if (do_roots_n2 || do_roots_n3)
		{	
			omp_set_num_threads(ncore);
			#pragma omp parallel
			{
			    std::vector<RootResult> local_n2_roots, local_n3_roots;
			
			    #pragma omp for schedule(dynamic)
			    for (int ire = 0; ire <= Nre; ++ire) {
			        
					double re = re_min + ire * dre;
			        #pragma omp critical
			        {
			            std::cout << "Real: " << re << "\n";
			        }
			        for (int iim = 0; iim <= Nim; ++iim) {
			
			            double im = im_min + iim * dim;
		
                		if (do_roots_n2) {
                		    RootResult r2 =
                		        FindComplexZero_noDerivative(e2_bootstrap, re, im);

                		    if (IsGoodRoot(r2, max_found_root_size)) {
                		        local_n2_roots.push_back(r2);
                		    }
                		}

                		if (do_roots_n3) {
                		    RootResult r3 =
                		        FindComplexZero_noDerivative(e3_bootstrap, re, im);

                		    if (IsGoodRoot(r3, max_found_root_size)) {
                		        local_n3_roots.push_back(r3);
                		    }
                		}

			            // RootResult r = FindComplexZero_noDerivative(e2_bootstrap, re, im);
			
						// if (!r.ok) continue;
						// if (r.status != 0) continue;
						// if (!std::isfinite(r.k.real()) || !std::isfinite(r.k.imag())) continue;
						// if (std::abs(r.F) > 1e-8) continue;

						// const double axis_tol = 1e-8;

						// if (r.k.real() < -axis_tol) continue;
						// if (r.k.imag() < -axis_tol) continue;
						// 
						// if (std::abs(r.k.real()) < axis_tol) r.k.real(0.0);
						// if (std::abs(r.k.imag()) < axis_tol) r.k.imag(0.0);

						// if (std::abs(r.k) > max_found_root_size)
    					// 	continue;
			            // local_roots.push_back(r);
			
			        }
			    }
			
			    // #pragma omp critical
			    // {
			    //     roots.insert(roots.end(), local_roots.begin(), local_roots.end());
			    // }
				#pragma omp critical
        		{
        		    roots_n2.insert(
        		        roots_n2.end(),
        		        local_n2_roots.begin(),
        		        local_n2_roots.end()
        		    );

        		    roots_n3.insert(
        		        roots_n3.end(),
        		        local_n3_roots.begin(),
        		        local_n3_roots.end()
        		    );
        		}
			}

			const double tol = 1e-6;
			
			std::vector<RootResult> unique_roots_n2, unique_roots_n3;
			
			for (const auto& r : roots_n2) {
			
			    bool duplicate = false;
			
			    for (const auto& u : unique_roots_n2) {
			
			        if (std::abs(r.k - u.k) < tol) {
			            duplicate = true;
			            break;
			        }
			    }
			
			    if (!duplicate) {
			        unique_roots_n2.push_back(r);
			    }
			}
			for (const auto& r : roots_n3) {
			
				bool duplicate = false;
			
				for (const auto& u : unique_roots_n3) {
			
					if (std::abs(r.k - u.k) < tol) {
						duplicate = true;
						break;
					}
				}
			
				if (!duplicate) {
					unique_roots_n3.push_back(r);
				}
			}

			std::sort(
			    unique_roots_n2.begin(),
			    unique_roots_n2.end(),
			    [](const RootResult& a, const RootResult& b) {
			        return std::abs(a.k) < std::abs(b.k);
			    }
			);
			if (do_roots_n2)
				roots_n2_from_resampling.push_back(unique_roots_n2);
			
			std::sort(
			    unique_roots_n3.begin(),
			    unique_roots_n3.end(),
			    [](const RootResult& a, const RootResult& b) {
			        return std::abs(a.k) < std::abs(b.k);
			    }
			);
			if (do_roots_n3)
				roots_n3_from_resampling.push_back(unique_roots_n3);
		};
	}


    // Output file
	//
	//
	std::filesystem::create_directories(output_folder);
    
	
	if (do_roots_n2) {
	
	    std::string filename_n2 =
	        std::string(output_folder) + "/roots_n2.dat";
	
	    std::ofstream out_n2(filename_n2);
	
	    if (!out_n2.is_open()) {
	        std::cerr << "Cannot open " << filename_n2 << "\n";
	        return;
	    }
	
	    out_n2 << "# ires  iroot  Re(k)  Im(k)  Re(F)  Im(F)\n";
	
	    for (std::size_t ires = 0;
	         ires < roots_n2_from_resampling.size();
	         ++ires) {
	
	        const auto& roots = roots_n2_from_resampling[ires];
	
	        for (std::size_t iroot = 0;
	             iroot < roots.size();
	             ++iroot) {
	
	            const auto& r = roots[iroot];
	
	            out_n2
	                << ires << " "
	                << iroot << " "
	                << r.k.real() << " "
	                << r.k.imag() << " "
	                << r.F.real() << " "
	                << r.F.imag() << "\n";
	        }
	    }
	
	    out_n2.close();
	}	
		
		
	if (do_roots_n3) {
	
	    std::string filename_n3 =
	        std::string(output_folder) + "/roots_n3.dat";
	
	    std::ofstream out_n3(filename_n3);
	
	    if (!out_n3.is_open()) {
	        std::cerr << "Cannot open " << filename_n3 << "\n";
	        return;
	    }
	
	    out_n3 << "# ires  iroot  Re(k)  Im(k)  Re(F)  Im(F)\n";
	
	    for (std::size_t ires = 0;
	         ires < roots_n3_from_resampling.size();
	         ++ires) {
	
	        const auto& roots = roots_n3_from_resampling[ires];
	
	        for (std::size_t iroot = 0;
	             iroot < roots.size();
	             ++iroot) {
	
	            const auto& r = roots[iroot];
	
	            out_n3
	                << ires << " "
	                << iroot << " "
	                << r.k.real() << " "
	                << r.k.imag() << " "
	                << r.F.real() << " "
	                << r.F.imag() << "\n";
	        }
	    }
	
	    out_n3.close();
	}	


	if (do_cumulants_n2) {
	    std::ofstream out(std::string(output_folder) + "/cumulants_n2.dat");
	
	    out << "# ires  vn2to2  vn4to4  vn6to6  vn8to8  vn10to10\n";
	
	    for (std::size_t ires = 0; ires < cumulants_n2_from_resampling.size(); ++ires) {
	        const auto& c = cumulants_n2_from_resampling[ires];
	
	        out << ires << " "
	            << c.vn2to2 << " "
	            << c.vn4to4 << " "
	            << c.vn6to6 << " "
	            << c.vn8to8 << " "
	            << c.vn10to10 << "\n";
	    }
	}
	
	if (do_cumulants_n3) {
	    std::ofstream out(std::string(output_folder) + "/cumulants_n3.dat");
	
	    out << "# ires  vn2to2  vn4to4  vn6to6  vn8to8  vn10to10\n";
	
	    for (std::size_t ires = 0; ires < cumulants_n3_from_resampling.size(); ++ires) {
	        const auto& c = cumulants_n3_from_resampling[ires];
	
	        out << ires << " "
	            << c.vn2to2 << " "
	            << c.vn4to4 << " "
	            << c.vn6to6 << " "
	            << c.vn8to8 << " "
	            << c.vn10to10 << "\n";
	    }
	}
	// std::ofstream out(output_filename);

	// if (!out.is_open()) {
	//     std::cerr << "Cannot open output file\n";
	//     return;
	// }
	// 
	// out << "# ires  iroot  Re(k)  Im(k)  Re(F)  Im(F)\n";
	// 
	// for (std::size_t ires = 0;
	//      ires < roots_from_resampling.size();
	//      ++ires) {
	// 
	//     const auto& roots = roots_from_resampling[ires];
	// 
	//     for (std::size_t iroot = 0;
	//          iroot < roots.size();
	//          ++iroot) {
	// 
	//         const auto& r = roots[iroot];
	// 
	//         out
	//             << ires << " "
	//             << iroot << " "
	//             << r.k.real() << " "
	//             << r.k.imag() << " "
	//             << r.F.real() << " "
	//             << r.F.imag() << "\n";
	//     }
	// }
	// 
	// out.close();

	// Fill a 2D histogram
	// TH2D* h_roots = new TH2D(
	//     "h_roots",
	//     "Bootstrap roots;Re(k);Im(k)",
	//     200, 0, 0.7 * max_found_root_size,
	//     200, 0, 1.1 * im_max
	// );
	// 
	// for (const auto& roots : roots_from_resampling) {
	//     for (const auto& r : roots) {
	//         h_roots->Fill(r.k.real(), r.k.imag());
	//     }
	// }

	// ---> Plot roots histograms
	// -------------------------
	// 
	// TFile outFile("bootstrap_roots_hist.root", "RECREATE");
	// h_roots->Write();
	// outFile.Close();

	// TCanvas* c = new TCanvas("c_roots", "Bootstrap roots", 800, 700);
	// h_roots->Draw("COLZ");
	// c->SaveAs("bootstrap_roots_hist.pdf");
	
	
    // ---> Plot G(k)
	//------------------
	//
	// TGraph* gr = new TGraph();

    // int point = 0;
    // for (double k = 0.0; k <= 200.0; k += 0.5) {
    //     double val = 0.0;
    //     MeanJ0(k, e2_list, val);
    //     gr->SetPoint(point, k, val);
    //     ++point;
    // }

    // TCanvas* c = new TCanvas("c", "J0 average", 800, 600);
    // c->SetGrid();

    // gr->SetTitle("<J_{0}(k #epsilon_{2})>;k;<J_{0}>");
    // gr->SetLineWidth(2);
    // gr->GetYaxis()->SetRangeUser(-0.01, 0.01);
    // gr->Draw("AL");

    // c->SaveAs("J0_vs_k.pdf");
}
