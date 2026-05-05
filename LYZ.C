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
#include <Math/IFunction.h>

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



class ReMeanJ0Grad : public ROOT::Math::IGradientFunctionMultiDim {
public:
    ReMeanJ0Grad(const std::vector<double>& e2) : e2_vals(e2) {}

    unsigned int NDim() const override { return 2; }

    ROOT::Math::IBaseFunctionMultiDim* Clone() const override {
        return new ReMeanJ0Grad(*this);
    }

private:
    double DoEval(const double* x) const override {
        std::complex<double> k{x[0], x[1]};
        std::complex<double> val;
        MeanJ0_complex(k, e2_vals, val);
        return val.real();
    }

    double DoDerivative(const double* x, unsigned int icoord) const override {
        std::complex<double> k{x[0], x[1]};
        std::complex<double> dfdk;
        MeanJ0_derivative_complex(k, e2_vals, dfdk);

        double a = dfdk.real();
        double b = dfdk.imag();

        return (icoord == 0) ? a : -b;
    }

    const std::vector<double>& e2_vals;
};

class ImMeanJ0Grad : public ROOT::Math::IGradientFunctionMultiDim {
public:
    ImMeanJ0Grad(const std::vector<double>& e2) : e2_vals(e2) {}

    unsigned int NDim() const override { return 2; }

    ROOT::Math::IBaseFunctionMultiDim* Clone() const override {
        return new ImMeanJ0Grad(*this);
    }

private:
    double DoEval(const double* x) const override {
        std::complex<double> k{x[0], x[1]};
        std::complex<double> val;
        MeanJ0_complex(k, e2_vals, val);
        return val.imag();
    }

    double DoDerivative(const double* x, unsigned int icoord) const override {
        std::complex<double> k{x[0], x[1]};
        std::complex<double> dfdk;
        MeanJ0_derivative_complex(k, e2_vals, dfdk);

        double a = dfdk.real();
        double b = dfdk.imag();

        return (icoord == 0) ? b : a;
    }

    const std::vector<double>& e2_vals;
};


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
	
	int max_events = 10000;
	int counter = 0;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') continue;
		
		counter ++;
		if (counter == max_events) break;
        
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



	// Find complex zeros ---> Method does not need derivative
	ROOT::Math::GSLMultiRootFinder finder(ROOT::Math::GSLMultiRootFinder::kHybridS);

	ReMeanJ0 f_re(e2_list);
	ImMeanJ0 f_im(e2_list);
        
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
	


	// Find complex zeros ---> Method DOES need derivative



	
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
