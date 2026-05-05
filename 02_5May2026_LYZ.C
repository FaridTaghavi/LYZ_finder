#include <cmath>
#include <complex>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "TCanvas.h"
#include "TGraph.h"
#include "TMath.h"
#include "TAxis.h"

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
// example of using multi root finder based on GSL
// need to use an algorithm not requiring the derivative
//like hybrids (default), hybrid, dnewton, broyden
 
using namespace ROOT::Math;


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



void MeanJ0_derivative_complex(std::complex<double> k,
                          const std::vector<double>& e2_vals,
                          std::complex<double>& result)
{
    std::complex<double> sum{0.0, 0.0};
    const std::size_t N = e2_vals.size();
    
	for (double e2 : e2_vals) {
        sum += -e2 * J1_complex(k * e2);
    }

    result = (N > 0) ?  sum / static_cast<double>(N) : std::complex<double>(0.0, 0.0);
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


// struct MeanJ0Functor {
//     const std::vector<double>& e2_vals;
// 
//     explicit MeanJ0Functor(const std::vector<double>& vals) : e2_vals(vals) {}
// 
//     double operator()(double k) const
//     {
//         double result = 0.0;
//         MeanJ0(k, e2_vals, result);
//         return result;
//     }
// };

struct MeanJ0ComplexFunctor {
    const std::vector<double>& e2_vals;

    explicit MeanJ0ComplexFunctor(const std::vector<double>& vals) : e2_vals(vals) {}

    std::complex<double> operator()(std::complex<double> k) const
    {
        std::complex<double> result = 0.0;
        MeanJ0_complex(k, e2_vals, result);
        return result;
    }
};



// void find_intervals(const MeanJ0Functor& Gk,
//                     double k_max,
//                     double dk,
//                     std::vector<std::pair<double, double>>& intervals)
// {
//     if (k_max <= 0.0 || dk <= 0.0) {
//         throw std::invalid_argument("find_intervals requires k_max > 0 and dk > 0");
//     }
// 
//     intervals.clear();
// 
//     const std::size_t N = static_cast<std::size_t>(std::ceil(k_max / dk));
// 
//     double k0 = 0.0;
//     double f0 = Gk(k0);
// 
//     for (std::size_t i = 1; i <= N; ++i) {
//         const double k1 = std::min(static_cast<double>(i) * dk, k_max);
//         const double f1 = Gk(k1);
// 
//         if (f0 == 0.0) {
//             intervals.emplace_back(k0, k0);
//         } else if (f0 * f1 <= 0.0) {
//             intervals.emplace_back(k0, k1);
//         }
// 
//         k0 = k1;
//         f0 = f1;
//     }
// }

// std::vector<double> find_roots(const MeanJ0Functor& Gk,
//                                const std::vector<std::pair<double, double>>& intervals)
// {
//     std::vector<double> roots;
//     ROOT::Math::Functor1D f(Gk);
// 
//     for (const auto& interval : intervals) {
//         const double a = interval.first;
//         const double b = interval.second;
// 
//         if (a == b) {
//             roots.push_back(a);
//             std::cout << "root = " << a << "   G(root) = " << Gk(a) << "\n";
//             continue;
//         }
// 
//         ROOT::Math::RootFinder solver(ROOT::Math::RootFinder::kBRENT);
//         solver.SetFunction(f, a, b);
// 
//         if (solver.Solve()) {
//             const double root = solver.Root();
//             roots.push_back(root);
//             std::cout << "root = " << root << "   G(root) = " << Gk(root) << "\n";
//         } else {
//             std::cerr << "Root finding failed in interval [" << a << ", " << b << "]\n";
//         }
//     }
// 
//     return roots;
// }

void LYZ(const char* filename = "PbPb_central_4.dat")
{
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: cannot open file " << filename << std::endl;
        return;
    }

    double id, b, npart, multi;
    double ex2, ex3, ex4, ex5;
    double ey2, ey3, ey4, ey5;
    double r2, r3, r4, r5;

    std::vector<double> e2_list;
    e2_list.reserve(1000000);

    std::string line;
	
	// int max_events = 10000;
	// int counter = 0;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') continue;
		
		// counter ++;
		// if (counter == max_events) break;
        
		std::istringstream iss(line);

        if (!(iss >> id >> b >> npart >> multi
                  >> ex2 >> ex3 >> ex4 >> ex5
                  >> ey2 >> ey3 >> ey4 >> ey5
                  >> r2 >> r3 >> r4 >> r5)) {
            continue;
        }

        const double e2 = std::sqrt(ex2 * ex2 + ey2 * ey2);
        e2_list.push_back(e2);
    }

    std::cout << "Loaded " << e2_list.size() << " events\n";

    if (e2_list.empty()) {
        std::cerr << "No events were read. Check the input file format.\n";
        return;
    }

    // double av_J0 = 0.0;
    // MeanJ0(32.0, e2_list, av_J0);
    // std::cout << "Real G(32) = " << av_J0 << "\n";

    // const double k_max = 150.0;
    // const double dk = 1.0;

    // MeanJ0Functor Gk(e2_list);
    // std::vector<std::pair<double, double>> intervals;
    // find_intervals(Gk, k_max, dk, intervals);

    // std::cout << "Sign-change intervals:\n";
    // for (const auto& p : intervals) {
    //     std::cout << "[" << p.first << ", " << p.second << "]\n";
    // }

    // std::vector<double> roots = find_roots(Gk, intervals);
    // MeanJ0ComplexFunctor G_complex(e2_list);
    // const std::complex<double> k_comp(10.0, 1.0);
    // const std::complex<double> complex_value = G_complex(k_comp);

    // std::complex<double> complex_value_derivative;

	ReMeanJ0 f_re(e2_list);
	ImMeanJ0 f_im(e2_list);
	// double x[2] = {10.0, 1.0};
    
	// MeanJ0_derivative_complex(k_comp, e2_list, complex_value_derivative);
    // std::cout << "Complex G(10 + i) = " << complex_value << "   " <<  f_re(x)  << "   " <<  f_im(x)   <<  "\n";
    // std::cout << "Complex dG(10 + i) = " << complex_value_derivative << "\n";


	// Find complex zeros
	ROOT::Math::GSLMultiRootFinder finder(ROOT::Math::GSLMultiRootFinder::kHybridS);

        
    finder.AddFunction(f_re, 2);
    finder.AddFunction(f_im, 2);
	
	double x0[2] = {50.0, 10.0};
	
	bool ok = finder.Solve(x0, 1000, 1e-10, 1e-10);

    const double* root = finder.X();
    const double* fval = finder.FVal();

    std::cout << "success = " << ok << "\n";
    std::cout << "status  = " << finder.Status() << "\n";
    std::cout << "k = " << root[0] << " + i " << root[1] << "\n";
    std::cout << "F(k) = " << fval[0] << " , " << fval[1] << "\n";
	
	
	
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
