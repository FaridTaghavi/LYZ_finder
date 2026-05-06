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


void LYZ(const char* filename = "PbPb_events.root")
{
	
	
	// Searching the root in (ReZ, ImZ) plane
	const double re_min {30.0};
	const double re_max {80.0};
	const double dre    {15.0};
	const double im_min {0.0};
	const double im_max {30};
	const double dim    {10.0};
    
	const double max_found_root_size = 2 * std::sqrt( re_max * re_max + im_max * im_max ); // sometimes the rootfinder finds very big roots, we discard them. 
	
	// Bootstrap
	const int Nsub = 10;  // Bootstrap subsample
	const int Nres = 500; // Bootstrap resampling time
	
	// Be careful to remove it later!
	const std::size_t max_events = 10000000;
	
	// Numebr of cores in searching for the roots
	const int ncore = 13;
	
	
	TFile file(filename, "READ");

    if (file.IsZombie()) {
        std::cerr << "Error: cannot open ROOT file "
                  << filename << "\n";
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
		

	std::vector<std::vector<double>> subsamples(Nsub);


	// Reading ttree and fill the subsample	
	for (std::size_t i = 0; i < nevents; ++i) {
	    tree->GetEntry(i);
	
	    double ex2 = event->Get_epsilonx(2);
	    double ey2 = event->Get_epsilony(2);
	
	    double e2 = std::sqrt(ex2 * ex2 + ey2 * ey2);
	    
		// e2_list.push_back(e2); //Not needed anymore!

    	size_t isub = i % Nsub;
    	subsamples[isub].push_back(e2);
	}
	
	
	std::mt19937 rng(12345);
	std::uniform_int_distribution<int> dist(0, Nsub - 1);
	
	std::vector<std::vector<RootResult>> roots_from_resampling;
	
	for (int ires = 0; ires < Nres; ++ires) {
	
	    std::vector<double> e2_bootstrap;
	    e2_bootstrap.reserve(nevents);
	
	    for (int isub = 0; isub < Nsub; ++isub) {
	        int pick = dist(rng);
	
	        e2_bootstrap.insert(
	            e2_bootstrap.end(),
	            subsamples[pick].begin(),
	            subsamples[pick].end()
	        );
	    }
		std::cout << "----------------------------------\n";	
    	std::cout << "Bootstrap sample " << ires + 1
    	          << " out of " << Nres
    	          << " resampling.\n";	
		std::cout << "----------------------------------\n";	
	
		std::vector<RootResult> roots;
	
	
		int Nre = static_cast<int>((re_max - re_min) / dre);
		int Nim = static_cast<int>((im_max - im_min) / dim);

		// Turn off the root finder warning info
		gsl_set_error_handler_off();
		gErrorIgnoreLevel = kWarning;
		gErrorIgnoreLevel = kFatal;	
		
		omp_set_num_threads(ncore);
		#pragma omp parallel
		{
		    std::vector<RootResult> local_roots;
		
		    #pragma omp for schedule(dynamic)
		    for (int ire = 0; ire <= Nre; ++ire) {
		        
				double re = re_min + ire * dre;
		        #pragma omp critical
		        {
		            std::cout << "Real: " << re << "\n";
		        }
		        for (int iim = 0; iim <= Nim; ++iim) {
		
		            double im = im_min + iim * dim;
		
		            RootResult r = FindComplexZero_noDerivative(e2_bootstrap, re, im);
		
					if (!r.ok) continue;
					if (r.status != 0) continue;
					if (!std::isfinite(r.k.real()) || !std::isfinite(r.k.imag())) continue;
					if (std::abs(r.F) > 1e-8) continue;

					const double axis_tol = 1e-8;

					if (r.k.real() < -axis_tol) continue;
					if (r.k.imag() < -axis_tol) continue;
					
					if (std::abs(r.k.real()) < axis_tol) r.k.real(0.0);
					if (std::abs(r.k.imag()) < axis_tol) r.k.imag(0.0);

					if (std::abs(r.k) > max_found_root_size)
    					continue;
		            local_roots.push_back(r);
		
		        }
		    }
		
		    #pragma omp critical
		    {
		        roots.insert(roots.end(), local_roots.begin(), local_roots.end());
		    }
		}

		std::vector<RootResult> unique_roots;
		
		const double tol = 1e-6;
		
		for (const auto& r : roots) {
		
		    bool duplicate = false;
		
		    for (const auto& u : unique_roots) {
		
		        if (std::abs(r.k - u.k) < tol) {
		            duplicate = true;
		            break;
		        }
		    }
		
		    if (!duplicate) {
		        unique_roots.push_back(r);
		    }
		}

		std::sort(
		    unique_roots.begin(),
		    unique_roots.end(),
		    [](const RootResult& a, const RootResult& b) {
		        return std::abs(a.k) < std::abs(b.k);
		    }
		);
		roots_from_resampling.push_back(unique_roots);
	}


	// Write in a .dat file
	std::ofstream out("bootstrap_roots.dat");
	
	if (!out.is_open()) {
	    std::cerr << "Cannot open output file\n";
	    return;
	}
	
	out << "# ires  iroot  Re(k)  Im(k)  Re(F)  Im(F)\n";
	
	for (std::size_t ires = 0;
	     ires < roots_from_resampling.size();
	     ++ires) {
	
	    const auto& roots = roots_from_resampling[ires];
	
	    for (std::size_t iroot = 0;
	         iroot < roots.size();
	         ++iroot) {
	
	        const auto& r = roots[iroot];
	
	        out
	            << ires << " "
	            << iroot << " "
	            << r.k.real() << " "
	            << r.k.imag() << " "
	            << r.F.real() << " "
	            << r.F.imag() << "\n";
	    }
	}
	
	out.close();

	// Fill a 2D histogram
	TH2D* h_roots = new TH2D(
	    "h_roots",
	    "Bootstrap roots;Re(k);Im(k)",
	    200, 0, 0.7 * max_found_root_size,
	    200, 0, 1.1 * im_max
	);
	
	for (const auto& roots : roots_from_resampling) {
	    for (const auto& r : roots) {
	        h_roots->Fill(r.k.real(), r.k.imag());
	    }
	}


	TFile outFile("bootstrap_roots_hist.root", "RECREATE");
	h_roots->Write();
	outFile.Close();

	TCanvas* c = new TCanvas("c_roots", "Bootstrap roots", 800, 700);
	h_roots->Draw("COLZ");
	c->SaveAs("bootstrap_roots_hist.pdf");
	
	// std::cout << "Loaded " << e2_list.size()
    //           << " events from ROOT file\n";

    // if (e2_list.empty()) {
    //     std::cerr << "No events were read.\n";
    //     return;
    // }


	// Find complex zeros ---> Method does not need derivative


	// RootResult r = FindComplexZero_noDerivative(e2_list, 50.0, 10.0);
	// 
	// std::cout << "success = " << r.ok << "\n";
	// std::cout << "status  = " << r.status << "\n";
	// std::cout << "k = " << r.k.real() << " + i " << r.k.imag() << "\n";
	// std::cout << "F(k) = " << r.F.real() << " , " << r.F.imag() << "\n";
	// std::cout << "time = " << r.time_ms << " ms\n";



	// for (double re = re_min; re <= re_max; re += dre) {
	//     for (double im = im_min; im <= im_max; im += dim) {
	// 
	//         RootResult r = FindComplexZero_noDerivative(e2_list, re, im);
	// 
	//         if (!r.ok) continue;
	// 
	//         if (std::abs(r.F) > 1e-8) continue;
	// 
	//         roots.push_back(r);
	// 		std::cout << "Real: " << re << ", Imeaginary: " << im << "\n";
	//     }
	// }

	// Print ---
	// for (const auto& r : unique_roots) {
	// 
	//     std::cout
	//         << "k = "
	//         << r.k.real()
	//         << " + i "
	//         << r.k.imag()
	//         << "\n";
	// 
	//     // std::cout
	//     //     << "F(k) = "
	//     //     << r.F.real()
	//     //     << " + i "
	//     //     << r.F.imag()
	//     //     << "\n";
	// }
	
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
