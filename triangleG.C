#include <TROOT.h>
#include <TStyle.h>
#include <TRandom3.h>
#include <TGraph.h>
#include <TCanvas.h>
#include <TLegend.h>
#include <TAxis.h>
#include <TMath.h>
#include <TError.h>

#include <Math/GSLMultiRootFinder.h>
#include <Math/IFunction.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_eigen.h>
#include <omp.h>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <sstream>
#include <utility>
#include <filesystem>

using cd = std::complex<double>;

struct TriangleGConfig {
    double a = 0.04;
    double b = 0.15;

    int M = 300;
    int M_flow = -1;
    int Nevents = 2000000;
    int Nsub = 30;
    int Nresam = 100;
    int ncore = 1;
    int q_bins = 20000;
    int vn_bins = 2000;
    int nonflow_q = 0;
    int nonflow_groups = -1;
    int M_nonflow = -1;
    int Neta = 1;
    double nonflow_delta = 0.0;
    double nonflow_delta_eta = 0.0;
    double eta_max = 2.5;
    bool M_flow_set = false;
    bool M_nonflow_set = false;
    bool nonflow_groups_set = false;
    std::string generating_function = "exponential";
    int modified_gl_nodes = 8;
    int product_theta_bins = 1;

    int n = 2;
    double theta = M_PI / 3.0;

    double root_start_re = 0.175344;
    double root_start_im = 0.0542001;
    std::vector<std::pair<double, double>> root_start_points;

    unsigned int seed = 0;
    unsigned int bootstrap_seed = 12345;

    std::string nominal_output = "MC_generating_roots.dat";
    std::string bootstrap_output = "MC_generating_roots_bootstrap.dat";
    std::string output_folder;
    std::string root_diagnostics_output;
    std::string coefficient_error_output;
    std::string vn_error_output;
    std::string vn_error_source = "subsample";
    std::string vn_direct_error_source = "subsample";
    std::string vn_j0_error_source = "none";
    int coefficient_error_grid = 121;
    double coefficient_error_half_width = -1.0;
    double coefficient_error_re_min = NAN;
    double coefficient_error_re_max = NAN;
    double coefficient_error_im_min = NAN;
    double coefficient_error_im_max = NAN;
    std::string correlations_output = "correlations_2_4_6_8_10.dat";
    bool write_correlations = true;
    bool write_vn_lyz = true;
    bool write_vn_j0 = false;
    bool write_correlation_truncated_lyz = false;
};

void PrintTriangleGUsage(const char* program)
{
    std::cout
        << "Usage: " << program << " [options]\n\n"
        << "Options:\n"
        << "  --a VALUE                    Triangle mode parameter a\n"
        << "  --b VALUE                    Triangle max parameter b\n"
        << "  --M VALUE                    Legacy shortcut for --M-flow when --M-flow is not set\n"
        << "  --M-flow VALUE               Flow multiplicity; can be 0\n"
        << "  --N VALUE                    Number of generated events\n"
        << "  --Nevents VALUE              Same as --N\n"
        << "  --Nresam VALUE               Number of bootstrap resamples\n"
        << "  --Nsub VALUE                 Number of Q subsamples for bootstrap\n"
        << "  --ncore VALUE                Number of OpenMP threads\n"
        << "  --q-bins VALUE               Bins for weighted Q root search; 0 is exact\n"
        << "  --vn-bins VALUE              Bins for direct sampled-vn LYZ root search\n"
        << "  --Neta VALUE                 Number of eta subevents in [-eta_max, eta_max]\n"
        << "  --M-nonflow VALUE            Total copied uniform nonflow multiplicity; default M\n"
        << "  --nonflow-q VALUE            Copy count q for uniform nonflow particles; 0 off\n"
        << "  --nonflow-delta-phi VALUE    Uniform smearing half-width around each nonflow cluster angle\n"
        << "  --nonflow-delta-eta VALUE    Uniform smearing half-width around each nonflow cluster eta\n"
        << "  --nonflow-groups VALUE       Independent nonflow particles; default M/q\n"
        << "  --eta-max VALUE              Assign each particle eta uniformly in [-eta_max, eta_max]\n"
        << "  --generating-function VALUE  Zero definition: exponential, product, product-falling, product-falling-subevents, product-correction, product-correction2, modified, exact, both, or all\n"
        << "  --modified-gl-nodes VALUE    N_GL nodes used for the equation-18 modified product\n"
        << "  --product-theta-bins VALUE   Uniform theta points averaged for product-based modes\n"
        << "  --root-start-re VALUE        Initial Re(k) for root search\n"
        << "  --root-start-im VALUE        Initial Im(k) for root search\n"
        << "  --root-start-point RE,IM     Initial guess; can be repeated\n"
        << "  --n VALUE                    Harmonic n used in Q_n(theta)\n"
        << "  --theta VALUE                Reference angle theta\n"
        << "  --seed VALUE                 Event-generation RNG seed\n"
        << "  --bootstrap-seed VALUE       Bootstrap RNG seed\n"
        << "  --nominal-output FILE        Nominal root output file\n"
        << "  --bootstrap-output FILE      Bootstrap roots output file\n"
        << "  --output-folder DIR          Directory for output files\n"
        << "  --root-diagnostics-output FILE  Product bootstrap root diagnostics output; empty disables\n"
        << "  --coefficient-error-output FILE  Product coefficient chi2 scan output; empty disables\n"
        << "  --vn-error-output FILE       Direct-vn G(z) chi2 scan output; empty disables\n"
        << "  --vn-error-source VALUE      Set both direct-vn chi2 covariance sources: none, subsample, or bootstrap\n"
        << "  --vn-direct-error-source VALUE  Truncated-J0 chi2 covariance: none, subsample, or bootstrap\n"
        << "  --vn-j0-error-source VALUE   Full-J0 chi2 covariance: none, subsample, or bootstrap\n"
        << "  --coefficient-error-grid VALUE   Grid points per axis for coefficient chi2 scan\n"
        << "  --coefficient-error-half-width VALUE  Half-width around nominal root for coefficient chi2 scan\n"
        << "  --coefficient-error-re-min VALUE  Global coefficient chi2 scan minimum Re(z)\n"
        << "  --coefficient-error-re-max VALUE  Global coefficient chi2 scan maximum Re(z)\n"
        << "  --coefficient-error-im-min VALUE  Global coefficient chi2 scan minimum Im(z)\n"
        << "  --coefficient-error-im-max VALUE  Global coefficient chi2 scan maximum Im(z)\n"
        << "  --correlations-output FILE   Event-level <2>, <4>, <6>, <8>, and <10> output file\n"
        << "  --write-correlations VALUE   Write event-level correlation rows: 1 yes, 0 no\n"
        << "  --write-vn-lyz VALUE         Write direct sampled-vn LYZ roots: 1 yes, 0 no\n"
        << "  --write-vn-j0 VALUE          Write unscaled full <J0(k vn)> roots: 1 yes, 0 no\n"
        << "  --write-corr-lyz VALUE       Write truncated LYZ roots using <2m>, requires M<=10: 1 yes, 0 no\n"
        << "  --help                       Print this help\n";
}

TriangleGConfig ParseTriangleGArgs(int argc, char** argv)
{
    TriangleGConfig cfg;

    auto require_value = [&](int& i, const std::string& option) -> std::string {
        if(i + 1 >= argc)
            throw std::runtime_error("Missing value for " + option);
        return argv[++i];
    };

    auto parse_root_start_point = [](const std::string& text) {
        std::string normalized = text;
        for(char& c : normalized) {
            if(c == ':' || c == ';')
                c = ',';
        }

        std::stringstream ss(normalized);
        std::string re_text;
        std::string im_text;

        if(!std::getline(ss, re_text, ',') ||
           !std::getline(ss, im_text, ',') ||
           re_text.empty() ||
           im_text.empty()) {
            throw std::runtime_error(
                "Invalid --root-start-point '" + text + "'. Use RE,IM."
            );
        }

        std::string extra;
        if(std::getline(ss, extra, ',')) {
            throw std::runtime_error(
                "Invalid --root-start-point '" + text + "'. Use exactly RE,IM."
            );
        }

        return std::make_pair(std::stod(re_text), std::stod(im_text));
    };

    for(int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if(arg == "--help" || arg == "-h") {
            PrintTriangleGUsage(argv[0]);
            std::exit(0);
        } else if(arg == "--a") {
            cfg.a = std::stod(require_value(i, arg));
        } else if(arg == "--b") {
            cfg.b = std::stod(require_value(i, arg));
        } else if(arg == "--M") {
            cfg.M = std::stoi(require_value(i, arg));
        } else if(arg == "--M-flow") {
            cfg.M_flow = std::stoi(require_value(i, arg));
            cfg.M_flow_set = true;
        } else if(arg == "--N" || arg == "--Nevents") {
            cfg.Nevents = std::stoi(require_value(i, arg));
        } else if(arg == "--Nresam") {
            cfg.Nresam = std::stoi(require_value(i, arg));
        } else if(arg == "--Nsub") {
            cfg.Nsub = std::stoi(require_value(i, arg));
        } else if(arg == "--ncore") {
            cfg.ncore = std::stoi(require_value(i, arg));
        } else if(arg == "--q-bins") {
            cfg.q_bins = std::stoi(require_value(i, arg));
        } else if(arg == "--vn-bins") {
            cfg.vn_bins = std::stoi(require_value(i, arg));
        } else if(arg == "--Neta") {
            cfg.Neta = std::stoi(require_value(i, arg));
        } else if(arg == "--M-nonflow" || arg == "--N-nonflow") {
            cfg.M_nonflow = std::stoi(require_value(i, arg));
            cfg.M_nonflow_set = true;
        } else if(arg == "--nonflow-q") {
            cfg.nonflow_q = std::stoi(require_value(i, arg));
        } else if(arg == "--nonflow-delta-phi" || arg == "--nonflow-delta") {
            cfg.nonflow_delta = std::stod(require_value(i, arg));
        } else if(arg == "--nonflow-delta-eta" || arg == "--nonflow-deta-eta") {
            cfg.nonflow_delta_eta = std::stod(require_value(i, arg));
        } else if(arg == "--nonflow-groups") {
            cfg.nonflow_groups = std::stoi(require_value(i, arg));
            cfg.nonflow_groups_set = true;
        } else if(arg == "--eta-max") {
            cfg.eta_max = std::stod(require_value(i, arg));
        } else if(arg == "--generating-function") {
            cfg.generating_function = require_value(i, arg);
            if(cfg.generating_function == "exp")
                cfg.generating_function = "exponential";
            if(cfg.generating_function == "corrected-product")
                cfg.generating_function = "product-correction";
            if(cfg.generating_function == "corrected-product2")
                cfg.generating_function = "product-correction2";
            if(cfg.generating_function == "falling-product")
                cfg.generating_function = "product-falling";
            if(cfg.generating_function == "falling-subevents")
                cfg.generating_function = "product-falling-subevents";
        } else if(arg == "--modified-gl-nodes") {
            cfg.modified_gl_nodes = std::stoi(require_value(i, arg));
        } else if(arg == "--product-theta-bins") {
            cfg.product_theta_bins = std::stoi(require_value(i, arg));
        } else if(arg == "--root-start-re") {
            cfg.root_start_re = std::stod(require_value(i, arg));
        } else if(arg == "--root-start-im") {
            cfg.root_start_im = std::stod(require_value(i, arg));
        } else if(arg == "--root-start-point") {
            cfg.root_start_points.push_back(
                parse_root_start_point(require_value(i, arg))
            );
        } else if(arg == "--n") {
            cfg.n = std::stoi(require_value(i, arg));
        } else if(arg == "--theta") {
            cfg.theta = std::stod(require_value(i, arg));
        } else if(arg == "--seed") {
            cfg.seed = static_cast<unsigned int>(std::stoul(require_value(i, arg)));
        } else if(arg == "--bootstrap-seed") {
            cfg.bootstrap_seed =
                static_cast<unsigned int>(std::stoul(require_value(i, arg)));
        } else if(arg == "--nominal-output") {
            cfg.nominal_output = require_value(i, arg);
        } else if(arg == "--bootstrap-output") {
            cfg.bootstrap_output = require_value(i, arg);
        } else if(arg == "--output-folder") {
            cfg.output_folder = require_value(i, arg);
        } else if(arg == "--root-diagnostics-output") {
            cfg.root_diagnostics_output = require_value(i, arg);
        } else if(arg == "--coefficient-error-output") {
            cfg.coefficient_error_output = require_value(i, arg);
        } else if(arg == "--vn-error-output") {
            cfg.vn_error_output = require_value(i, arg);
        } else if(arg == "--vn-error-source") {
            cfg.vn_error_source = require_value(i, arg);
            cfg.vn_direct_error_source = cfg.vn_error_source;
            cfg.vn_j0_error_source = cfg.vn_error_source;
        } else if(arg == "--vn-direct-error-source") {
            cfg.vn_direct_error_source = require_value(i, arg);
        } else if(arg == "--vn-j0-error-source") {
            cfg.vn_j0_error_source = require_value(i, arg);
        } else if(arg == "--coefficient-error-grid") {
            cfg.coefficient_error_grid = std::stoi(require_value(i, arg));
        } else if(arg == "--coefficient-error-half-width") {
            cfg.coefficient_error_half_width = std::stod(require_value(i, arg));
        } else if(arg == "--coefficient-error-re-min") {
            cfg.coefficient_error_re_min = std::stod(require_value(i, arg));
        } else if(arg == "--coefficient-error-re-max") {
            cfg.coefficient_error_re_max = std::stod(require_value(i, arg));
        } else if(arg == "--coefficient-error-im-min") {
            cfg.coefficient_error_im_min = std::stod(require_value(i, arg));
        } else if(arg == "--coefficient-error-im-max") {
            cfg.coefficient_error_im_max = std::stod(require_value(i, arg));
        } else if(arg == "--correlations-output" || arg == "--q-vector-output") {
            cfg.correlations_output = require_value(i, arg);
        } else if(arg == "--write-correlations") {
            const int value = std::stoi(require_value(i, arg));
            if(value != 0 && value != 1)
                throw std::runtime_error("--write-correlations must be 0 or 1");
            cfg.write_correlations = (value != 0);
        } else if(arg == "--write-vn-lyz") {
            const int value = std::stoi(require_value(i, arg));
            if(value != 0 && value != 1)
                throw std::runtime_error("--write-vn-lyz must be 0 or 1");
            cfg.write_vn_lyz = (value != 0);
        } else if(arg == "--write-vn-j0") {
            const int value = std::stoi(require_value(i, arg));
            if(value != 0 && value != 1)
                throw std::runtime_error("--write-vn-j0 must be 0 or 1");
            cfg.write_vn_j0 = (value != 0);
        } else if(arg == "--write-corr-lyz") {
            const int value = std::stoi(require_value(i, arg));
            if(value != 0 && value != 1)
                throw std::runtime_error("--write-corr-lyz must be 0 or 1");
            cfg.write_correlation_truncated_lyz = (value != 0);
        } else {
            throw std::runtime_error("Unknown option: " + arg);
        }
    }

    if(cfg.a <= 0.0) throw std::runtime_error("--a must be positive");
    if(cfg.b <= cfg.a) throw std::runtime_error("--b must be larger than --a");
    if(cfg.M <= 0) throw std::runtime_error("--M must be positive");
    if(cfg.M_flow < -1)
        throw std::runtime_error("--M-flow must be -1 or non-negative");
    if(cfg.Nevents <= 0) throw std::runtime_error("--N/--Nevents must be positive");
    if(cfg.Nsub <= 0) throw std::runtime_error("--Nsub must be positive");
    if(cfg.Nresam < 0) throw std::runtime_error("--Nresam must be non-negative");
    if(cfg.ncore <= 0) throw std::runtime_error("--ncore must be positive");
    if(cfg.q_bins < 0) throw std::runtime_error("--q-bins must be non-negative");
    if(cfg.vn_bins <= 0) throw std::runtime_error("--vn-bins must be positive");
    if(cfg.Neta <= 0) throw std::runtime_error("--Neta must be positive");
    if(cfg.M_nonflow < -1)
        throw std::runtime_error("--M-nonflow must be -1 or non-negative");
    if(cfg.nonflow_q < 0) throw std::runtime_error("--nonflow-q must be non-negative");
    if(cfg.nonflow_delta < 0.0)
        throw std::runtime_error("--nonflow-delta-phi must be non-negative");
    if(cfg.nonflow_delta_eta < 0.0)
        throw std::runtime_error("--nonflow-delta-eta must be non-negative");
    if(cfg.eta_max <= 0.0)
        throw std::runtime_error("--eta-max must be positive");
    if(cfg.nonflow_groups < -1)
        throw std::runtime_error("--nonflow-groups must be -1 or non-negative");
    if(cfg.generating_function != "exponential" &&
       cfg.generating_function != "product" &&
       cfg.generating_function != "product-falling" &&
       cfg.generating_function != "product-falling-subevents" &&
       cfg.generating_function != "product-correction" &&
       cfg.generating_function != "product-correction2" &&
       cfg.generating_function != "modified" &&
       cfg.generating_function != "exact" &&
       cfg.generating_function != "both" &&
       cfg.generating_function != "all") {
        throw std::runtime_error(
            "--generating-function must be one of: exponential, product, product-falling, product-falling-subevents, product-correction, product-correction2, modified, exact, both, all"
        );
    }
    if(cfg.modified_gl_nodes <= 0)
        throw std::runtime_error("--modified-gl-nodes must be positive");
    if(cfg.product_theta_bins <= 0)
        throw std::runtime_error("--product-theta-bins must be positive");
    if(cfg.coefficient_error_grid < 3)
        throw std::runtime_error("--coefficient-error-grid must be at least 3");
    if(cfg.vn_error_source != "none" &&
       cfg.vn_error_source != "subsample" &&
       cfg.vn_error_source != "bootstrap")
        throw std::runtime_error("--vn-error-source must be none, subsample, or bootstrap");
    if(cfg.vn_direct_error_source != "none" &&
       cfg.vn_direct_error_source != "subsample" &&
       cfg.vn_direct_error_source != "bootstrap")
        throw std::runtime_error("--vn-direct-error-source must be none, subsample, or bootstrap");
    if(cfg.vn_j0_error_source != "none" &&
       cfg.vn_j0_error_source != "subsample" &&
       cfg.vn_j0_error_source != "bootstrap")
        throw std::runtime_error("--vn-j0-error-source must be none, subsample, or bootstrap");
    if(cfg.coefficient_error_half_width == 0.0 ||
       cfg.coefficient_error_half_width < -1.0)
        throw std::runtime_error("--coefficient-error-half-width must be positive or -1");
    const bool coefficient_error_has_any_bound =
        std::isfinite(cfg.coefficient_error_re_min) ||
        std::isfinite(cfg.coefficient_error_re_max) ||
        std::isfinite(cfg.coefficient_error_im_min) ||
        std::isfinite(cfg.coefficient_error_im_max);
    const bool coefficient_error_has_all_bounds =
        std::isfinite(cfg.coefficient_error_re_min) &&
        std::isfinite(cfg.coefficient_error_re_max) &&
        std::isfinite(cfg.coefficient_error_im_min) &&
        std::isfinite(cfg.coefficient_error_im_max);
    if(coefficient_error_has_any_bound && !coefficient_error_has_all_bounds)
        throw std::runtime_error("Use all four coefficient-error bounds: re-min, re-max, im-min, im-max");
    if(coefficient_error_has_all_bounds) {
        if(cfg.coefficient_error_re_max <= cfg.coefficient_error_re_min)
            throw std::runtime_error("--coefficient-error-re-max must be larger than --coefficient-error-re-min");
        if(cfg.coefficient_error_im_max <= cfg.coefficient_error_im_min)
            throw std::runtime_error("--coefficient-error-im-max must be larger than --coefficient-error-im-min");
    }
    if(cfg.nonflow_q == 0 &&
       (cfg.nonflow_groups > 0 || cfg.M_nonflow > 0)) {
        throw std::runtime_error(
            "Nonflow particles require --nonflow-q > 0"
        );
    }
    if(cfg.M_nonflow_set && cfg.nonflow_groups_set)
        throw std::runtime_error("Use only one of --M-nonflow or --nonflow-groups");

    if(cfg.nonflow_q > 0 &&
       cfg.M_nonflow >= 0 &&
       cfg.M_nonflow % cfg.nonflow_q != 0) {
        throw std::runtime_error("--M-nonflow must be divisible by --nonflow-q");
    }
    if(!cfg.M_flow_set &&
       cfg.nonflow_q > 0 &&
       cfg.M_nonflow < 0 &&
       cfg.nonflow_groups < 0 &&
       cfg.M % cfg.nonflow_q != 0) {
        throw std::runtime_error(
            "Default nonflow groups M/q requires M divisible by --nonflow-q. "
            "Use --M-nonflow or --nonflow-groups to set it explicitly."
        );
    }
    if(cfg.n <= 0) throw std::runtime_error("--n must be positive");

    if(cfg.root_start_points.empty())
        cfg.root_start_points.push_back({cfg.root_start_re, cfg.root_start_im});

    return cfg;
}

std::string OutputPathInFolder(const std::string& folder, const std::string& filename)
{
    if(folder.empty() || filename.empty())
        return filename;

    const std::filesystem::path path(filename);
    if(path.is_absolute())
        return filename;

    return (std::filesystem::path(folder) / path).string();
}

void ApplyOutputFolder(TriangleGConfig& cfg)
{
    if(cfg.output_folder.empty())
        return;

    std::filesystem::create_directories(cfg.output_folder);

    cfg.nominal_output =
        OutputPathInFolder(cfg.output_folder, cfg.nominal_output);
    cfg.bootstrap_output =
        OutputPathInFolder(cfg.output_folder, cfg.bootstrap_output);
    cfg.root_diagnostics_output =
        OutputPathInFolder(cfg.output_folder, cfg.root_diagnostics_output);
    cfg.coefficient_error_output =
        OutputPathInFolder(cfg.output_folder, cfg.coefficient_error_output);
    cfg.vn_error_output =
        OutputPathInFolder(cfg.output_folder, cfg.vn_error_output);
    cfg.correlations_output =
        OutputPathInFolder(cfg.output_folder, cfg.correlations_output);
}

struct MCGSample {
    std::vector<double> q;
    std::vector<double> weight;
    double total_weight = 0.0;
};

struct QBinLayout {
    int bins = 0;
    double q_min = 0.0;
    double q_max = 0.0;
    double width = 0.0;
};

struct MCGValue {
    cd G = {0.0, 0.0};
    cd dGdk = {0.0, 0.0};
};

struct ProductGSample {
    std::vector<cd> coeffs;
    double total_weight = 0.0;
};

QBinLayout BuildQBinLayout(int bins, int M)
{
    QBinLayout layout;
    layout.bins = bins;
    layout.q_min = -static_cast<double>(M);
    layout.q_max = static_cast<double>(M);
    layout.width =
        (layout.bins > 0)
            ? (layout.q_max - layout.q_min) / static_cast<double>(layout.bins)
            : 0.0;
    return layout;
}

int QBinIndex(double q, const QBinLayout& layout)
{
    int ibin = static_cast<int>((q - layout.q_min) / layout.width);
    if(ibin < 0) ibin = 0;
    if(ibin >= layout.bins) ibin = layout.bins - 1;
    return ibin;
}

MCGSample BuildExactQSample(const std::vector<double>& Q)
{
    MCGSample sample;
    sample.q = Q;
    sample.weight.assign(Q.size(), 1.0);
    sample.total_weight = static_cast<double>(Q.size());
    return sample;
}

MCGSample BuildBinnedQSample(
    const std::vector<double>& Q,
    const QBinLayout& layout
)
{
    std::vector<double> bin_weights(static_cast<std::size_t>(layout.bins), 0.0);

    for(double q : Q)
        bin_weights[static_cast<std::size_t>(QBinIndex(q, layout))] += 1.0;

    MCGSample sample;
    sample.q.reserve(static_cast<std::size_t>(layout.bins));
    sample.weight.reserve(static_cast<std::size_t>(layout.bins));
    sample.total_weight = static_cast<double>(Q.size());

    for(int ibin = 0; ibin < layout.bins; ++ibin) {
        const double weight = bin_weights[static_cast<std::size_t>(ibin)];
        if(weight == 0.0) continue;

        const double q_center =
            layout.q_min + (static_cast<double>(ibin) + 0.5) * layout.width;

        sample.q.push_back(q_center);
        sample.weight.push_back(weight);
    }

    return sample;
}

MCGSample BuildQSample(
    const std::vector<double>& Q,
    const QBinLayout& layout
)
{
    return (layout.bins > 0)
        ? BuildBinnedQSample(Q, layout)
        : BuildExactQSample(Q);
}

MCGValue EvaluateMCGeneratingFunction(
    cd k,
    const MCGSample& sample
)
{
    MCGValue value;
    if(sample.total_weight <= 0.0) return value;

    cd sum = 0.0;
    cd derivative_sum = 0.0;
    const cd I(0.0, 1.0);

    for(std::size_t i = 0; i < sample.q.size(); ++i) {
        const double q = sample.q[i];
        const double weight = sample.weight[i];
        const cd exponential = std::exp(I * k * q);

        sum += weight * exponential;
        derivative_sum += weight * I * q * exponential;
    }

    value.G = sum / sample.total_weight;
    value.dGdk = derivative_sum / sample.total_weight;
    return value;
}

class MCGCachedEvaluator {
public:
    explicit MCGCachedEvaluator(const MCGSample& sample)
        : sample_(sample) {}

    const MCGValue& Eval(const double* x) const {
        if(!cache_valid_ || x[0] != last_re_ || x[1] != last_im_) {
            last_re_ = x[0];
            last_im_ = x[1];
            cached_value_ =
                EvaluateMCGeneratingFunction(cd(last_re_, last_im_), sample_);
            cache_valid_ = true;
        }

        return cached_value_;
    }

private:
    const MCGSample& sample_;
    mutable bool cache_valid_ = false;
    mutable double last_re_ = 0.0;
    mutable double last_im_ = 0.0;
    mutable MCGValue cached_value_;
};

struct RootResult {
    bool ok = false;
    int status = -999;
    cd k = {NAN, NAN};
    cd F = {NAN, NAN};
    double time_ms = 0.0;
};

struct RootDiagnostic {
    int ires = -1;
    int iroot = -1;
    cd z_start = {NAN, NAN};
    cd G_start = {NAN, NAN};
    cd Gprime_start = {NAN, NAN};
    cd z_linearized = {NAN, NAN};
    RootResult tracked;
};

struct ReMCG {
    const MCGSample& sample;

    explicit ReMCG(const MCGSample& values)
        : sample(values) {}

    double operator()(const double* x) const {
        return std::real(
            EvaluateMCGeneratingFunction(cd(x[0], x[1]), sample).G
        );
    }
};

struct ImMCG {
    const MCGSample& sample;

    explicit ImMCG(const MCGSample& values)
        : sample(values) {}

    double operator()(const double* x) const {
        return std::imag(
            EvaluateMCGeneratingFunction(cd(x[0], x[1]), sample).G
        );
    }
};

class ReMCGGrad : public ROOT::Math::IGradientFunctionMultiDim {
public:
    explicit ReMCGGrad(const MCGCachedEvaluator& evaluator)
        : evaluator_(&evaluator) {}

    unsigned int NDim() const override { return 2; }

    ROOT::Math::IBaseFunctionMultiDim* Clone() const override {
        return new ReMCGGrad(*this);
    }

private:
    double DoEval(const double* x) const override {
        return std::real(evaluator_->Eval(x).G);
    }

    double DoDerivative(const double* x, unsigned int icoord) const override {
        cd dGdk = evaluator_->Eval(x).dGdk;
        return (icoord == 0) ? std::real(dGdk) : -std::imag(dGdk);
    }

    const MCGCachedEvaluator* evaluator_;
};

class ImMCGGrad : public ROOT::Math::IGradientFunctionMultiDim {
public:
    explicit ImMCGGrad(const MCGCachedEvaluator& evaluator)
        : evaluator_(&evaluator) {}

    unsigned int NDim() const override { return 2; }

    ROOT::Math::IBaseFunctionMultiDim* Clone() const override {
        return new ImMCGGrad(*this);
    }

private:
    double DoEval(const double* x) const override {
        return std::imag(evaluator_->Eval(x).G);
    }

    double DoDerivative(const double* x, unsigned int icoord) const override {
        cd dGdk = evaluator_->Eval(x).dGdk;
        return (icoord == 0) ? std::imag(dGdk) : std::real(dGdk);
    }

    const MCGCachedEvaluator* evaluator_;
};

RootResult FindMCGeneratingZero(
    const MCGSample& sample,
    double x0_re,
    double x0_im,
    bool use_jacobian = true
)
{
    RootResult result;

    try {
        auto t1 = std::chrono::high_resolution_clock::now();

        if(use_jacobian) {
            ROOT::Math::GSLMultiRootFinder finder(
                ROOT::Math::GSLMultiRootFinder::kHybridSJ
            );

            MCGCachedEvaluator evaluator(sample);
            ReMCGGrad f_re(evaluator);
            ImMCGGrad f_im(evaluator);
            finder.AddFunction(f_re);
            finder.AddFunction(f_im);

            double x0[2] = {x0_re, x0_im};
            bool ok = finder.Solve(x0, 1000, 1e-10, 1e-10);

            const double* root = finder.X();
            const double* fval = finder.FVal();

            result.ok = ok;
            result.status = finder.Status();
            if(root && fval) {
                result.k = {root[0], root[1]};
                result.F = {fval[0], fval[1]};
            }
        } else {
            ROOT::Math::GSLMultiRootFinder finder(
                ROOT::Math::GSLMultiRootFinder::kHybridS
            );

            ReMCG f_re(sample);
            ImMCG f_im(sample);
            finder.AddFunction(f_re, 2);
            finder.AddFunction(f_im, 2);

            double x0[2] = {x0_re, x0_im};
            bool ok = finder.Solve(x0, 1000, 1e-10, 1e-10);

            const double* root = finder.X();
            const double* fval = finder.FVal();

            result.ok = ok;
            result.status = finder.Status();
            if(root && fval) {
                result.k = {root[0], root[1]};
                result.F = {fval[0], fval[1]};
            }
        }

        auto t2 = std::chrono::high_resolution_clock::now();
        result.time_ms =
            std::chrono::duration<double, std::milli>(t2 - t1).count();
    }
    catch(const std::exception&) {
        result.ok = false;
        result.status = -999;
    }

    return result;
}

bool IsGoodMCGRoot(const RootResult& r, double max_root_size)
{
    if(!r.ok) return false;
    if(r.status != 0) return false;
    if(!std::isfinite(r.k.real()) || !std::isfinite(r.k.imag())) return false;
    if(std::abs(r.F) > 1e-6) return false;
    if(r.k.real() < -1e-8 || r.k.imag() < -1e-8) return false;
    (void)max_root_size;
    return true;
}

std::vector<RootResult> FindMCGeneratingZerosOnGrid(
    const MCGSample& sample,
    double re_min,
    double re_max,
    double dre,
    double im_min,
    double im_max,
    double dim,
    bool use_jacobian = true
)
{
    std::vector<RootResult> roots;
    const double max_root_size =
        2.0 * std::sqrt(re_max * re_max + im_max * im_max);

    for(double re = re_min; re <= re_max + 0.5 * dre; re += dre) {
        for(double im = im_min; im <= im_max + 0.5 * dim; im += dim) {
            RootResult r = FindMCGeneratingZero(sample, re, im, use_jacobian);
            if(IsGoodMCGRoot(r, max_root_size))
                roots.push_back(r);
        }
    }

    const double duplicate_tol = 1e-6;
    std::vector<RootResult> unique_roots;

    for(const auto& r : roots) {
        bool duplicate = false;
        for(const auto& u : unique_roots) {
            if(std::abs(r.k - u.k) < duplicate_tol) {
                duplicate = true;
                break;
            }
        }

        if(!duplicate)
            unique_roots.push_back(r);
    }

    std::sort(
        unique_roots.begin(),
        unique_roots.end(),
        [](const RootResult& a, const RootResult& b) {
            return std::abs(a.k) < std::abs(b.k);
        }
    );

    return unique_roots;
}

std::vector<RootResult> FindMCGeneratingZerosFromStarts(
    const MCGSample& sample,
    const std::vector<std::pair<double, double>>& start_points,
    bool use_jacobian = true
)
{
    std::vector<RootResult> roots;

    double max_root_size = 0.0;
    for(const auto& p : start_points) {
        max_root_size = std::max(
            max_root_size,
            std::sqrt(p.first * p.first + p.second * p.second)
        );
    }
    max_root_size *= 2.0;
    if(max_root_size <= 0.0)
        max_root_size = std::numeric_limits<double>::infinity();

    for(const auto& p : start_points) {
        RootResult r = FindMCGeneratingZero(sample, p.first, p.second, use_jacobian);
        if(IsGoodMCGRoot(r, max_root_size))
            roots.push_back(r);
    }

    const double duplicate_tol = 1e-6;
    std::vector<RootResult> unique_roots;

    for(const auto& r : roots) {
        bool duplicate = false;
        for(const auto& u : unique_roots) {
            if(std::abs(r.k - u.k) < duplicate_tol) {
                duplicate = true;
                break;
            }
        }

        if(!duplicate)
            unique_roots.push_back(r);
    }

    std::sort(
        unique_roots.begin(),
        unique_roots.end(),
        [](const RootResult& a, const RootResult& b) {
            return std::abs(a.k) < std::abs(b.k);
        }
    );

    return unique_roots;
}

std::vector<cd> BuildProductCoefficients(
    int n,
    double theta,
    const std::vector<double>& phiEvent
)
{
    std::vector<cd> coeffs;
    coeffs.reserve(phiEvent.size() + 1);
    coeffs.push_back(cd(1.0, 0.0));

    for(double phi : phiEvent) {
        const cd factor(0.0, std::cos(n * (phi - theta)));
        coeffs.push_back(cd(0.0, 0.0));

        for(std::size_t m = coeffs.size() - 1; m > 0; --m)
            coeffs[m] += coeffs[m - 1] * factor;
    }

    return coeffs;
}

std::vector<cd> BuildThetaAveragedProductCoefficients(
    int n,
    double theta_offset,
    int theta_bins,
    const std::vector<double>& phiEvent
)
{
    std::vector<cd> average(phiEvent.size() + 1, cd(0.0, 0.0));
    std::vector<cd> compensation(phiEvent.size() + 1, cd(0.0, 0.0));
    const double theta_step =
        2.0 * M_PI / (static_cast<double>(n) * theta_bins);

    for(int itheta = 0; itheta < theta_bins; ++itheta) {
        const std::vector<cd> coefficients = BuildProductCoefficients(
            n,
            theta_offset + itheta * theta_step,
            phiEvent
        );
        for(std::size_t order = 0; order < average.size(); ++order) {
            const cd corrected = coefficients[order] - compensation[order];
            const cd updated = average[order] + corrected;
            compensation[order] = (updated - average[order]) - corrected;
            average[order] = updated;
        }
    }

    const double inverse_bins = 1.0 / theta_bins;
    for(cd& coefficient : average)
        coefficient *= inverse_bins;

    return average;
}

std::vector<double> BuildModifiedProductFactors(int multiplicity, int gl_nodes)
{
    // Golub-Welsch quadrature for x^M exp(-x)/M!, obtained from x = M y
    // in Eq. (18). The normalized quadrature weights are v_0^2.
    gsl_matrix* jacobi = gsl_matrix_calloc(gl_nodes, gl_nodes);
    gsl_vector* eigenvalues = gsl_vector_alloc(gl_nodes);
    gsl_matrix* eigenvectors = gsl_matrix_alloc(gl_nodes, gl_nodes);
    gsl_eigen_symmv_workspace* workspace = gsl_eigen_symmv_alloc(gl_nodes);

    if(!jacobi || !eigenvalues || !eigenvectors || !workspace)
        throw std::runtime_error("Failed to allocate generalized Laguerre quadrature");

    const double alpha = static_cast<double>(multiplicity);
    for(int i = 0; i < gl_nodes; ++i) {
        gsl_matrix_set(jacobi, i, i, 2.0 * i + 1.0 + alpha);
        if(i + 1 < gl_nodes) {
            const double off_diagonal =
                std::sqrt((i + 1.0) * (i + 1.0 + alpha));
            gsl_matrix_set(jacobi, i, i + 1, off_diagonal);
            gsl_matrix_set(jacobi, i + 1, i, off_diagonal);
        }
    }

    const int status =
        gsl_eigen_symmv(jacobi, eigenvalues, eigenvectors, workspace);
    if(status != GSL_SUCCESS) {
        gsl_eigen_symmv_free(workspace);
        gsl_matrix_free(eigenvectors);
        gsl_vector_free(eigenvalues);
        gsl_matrix_free(jacobi);
        throw std::runtime_error("Failed to compute generalized Laguerre quadrature");
    }
    gsl_eigen_symmv_sort(eigenvalues, eigenvectors, GSL_EIGEN_SORT_VAL_ASC);

    std::vector<double> factors(
        static_cast<std::size_t>(multiplicity + 1),
        0.0
    );
    for(int inode = 0; inode < gl_nodes; ++inode) {
        const double x = gsl_vector_get(eigenvalues, inode);
        const double first_component = gsl_matrix_get(eigenvectors, 0, inode);
        const double normalized_weight = first_component * first_component;
        const double scale = static_cast<double>(multiplicity) / x;
        double scale_power = 1.0;

        for(int order = 0; order <= multiplicity; ++order) {
            factors[static_cast<std::size_t>(order)] +=
                normalized_weight * scale_power;
            scale_power *= scale;
        }
    }

    gsl_eigen_symmv_free(workspace);
    gsl_matrix_free(eigenvectors);
    gsl_vector_free(eigenvalues);
    gsl_matrix_free(jacobi);
    return factors;
}

std::vector<double> BuildExactProductFactors(int multiplicity)
{
    std::vector<double> factors(
        static_cast<std::size_t>(multiplicity + 1),
        1.0
    );

    for(int order = 1; order <= multiplicity; ++order) {
        factors[static_cast<std::size_t>(order)] =
            factors[static_cast<std::size_t>(order - 1)] *
            static_cast<double>(multiplicity) /
            static_cast<double>(multiplicity - order + 1);

        if(!std::isfinite(factors[static_cast<std::size_t>(order)])) {
            throw std::runtime_error(
                "Exact recursive correction overflowed; use modified mode for this multiplicity"
            );
        }
    }

    return factors;
}

std::vector<double> BuildProductDerivativeCorrectionFactors(int multiplicity)
{
    std::vector<double> factors(
        static_cast<std::size_t>(multiplicity + 1),
        1.0
    );

    const double inverse_multiplicity =
        1.0 / static_cast<double>(multiplicity);
    for(int order = 0; order <= multiplicity; ++order) {
        const double p = static_cast<double>(order);
        factors[static_cast<std::size_t>(order)] =
            1.0 + 0.5 * p * (p - 1.0) * inverse_multiplicity;
    }

    return factors;
}

std::vector<double> BuildProductDerivativeCorrection2Factors(int multiplicity)
{
    std::vector<double> factors(
        static_cast<std::size_t>(multiplicity + 1),
        1.0
    );

    const double inverse_multiplicity =
        1.0 / static_cast<double>(multiplicity);
    const double inverse_multiplicity2 =
        inverse_multiplicity * inverse_multiplicity;
    for(int order = 0; order <= multiplicity; ++order) {
        const double p = static_cast<double>(order);
        const double first_correction =
            0.5 * p * (p - 1.0) * inverse_multiplicity;
        const double second_derivative = p * (p + 1.0);
        const double third_derivative =
            -p * (p + 1.0) * (p + 2.0);
        const double fourth_derivative =
            p * (p + 1.0) * (p + 2.0) * (p + 3.0);

        const double second_correction =
            (
                second_derivative +
                (5.0 / 6.0) * third_derivative +
                0.125 * fourth_derivative
            ) *
            inverse_multiplicity2;

        factors[static_cast<std::size_t>(order)] =
            1.0 + first_correction + second_correction;
    }

    return factors;
}

std::vector<cd> BuildModifiedProductCoefficients(
    const std::vector<cd>& product_coeffs,
    const std::vector<double>& modified_factors
)
{
    std::vector<cd> coeffs = product_coeffs;
    if(coeffs.size() != modified_factors.size())
        throw std::runtime_error("Modified-product factor size does not match multiplicity");

    for(std::size_t order = 0; order < coeffs.size(); ++order)
        coeffs[order] *= modified_factors[order];

    return coeffs;
}

void TransformProductSample(
    ProductGSample& sample,
    const std::vector<double>& coefficient_factors,
    bool remove_odd_coefficients
)
{
    if(!coefficient_factors.empty() &&
       coefficient_factors.size() != sample.coeffs.size()) {
        throw std::runtime_error("Polynomial correction factor size does not match multiplicity");
    }

    for(std::size_t order = 0; order < sample.coeffs.size(); ++order) {
        if(remove_odd_coefficients && order % 2 == 1) {
            sample.coeffs[order] = cd(0.0, 0.0);
        } else if(!coefficient_factors.empty()) {
            sample.coeffs[order] *= coefficient_factors[order];
        }
    }
}

void AddProductCoefficientsCompensated(
    std::vector<cd>& sum_coeffs,
    std::vector<cd>& compensation,
    const std::vector<cd>& coefficients
)
{
    for(std::size_t i = 0; i < coefficients.size(); ++i) {
        const cd corrected = coefficients[i] - compensation[i];
        const cd updated = sum_coeffs[i] + corrected;
        compensation[i] = (updated - sum_coeffs[i]) - corrected;
        sum_coeffs[i] = updated;
    }
}

ProductGSample BuildProductSampleFromSubsamples(
    const std::vector<std::vector<cd>>& subsample_coeff_sums,
    const std::vector<double>& subsample_counts
)
{
    ProductGSample sample;
    if(subsample_coeff_sums.empty())
        return sample;

    sample.coeffs.assign(subsample_coeff_sums.front().size(), cd(0.0, 0.0));
    std::vector<cd> compensation(sample.coeffs.size(), cd(0.0, 0.0));

    for(std::size_t isub = 0; isub < subsample_coeff_sums.size(); ++isub) {
        AddProductCoefficientsCompensated(
            sample.coeffs,
            compensation,
            subsample_coeff_sums[isub]
        );

        sample.total_weight += subsample_counts[isub];
    }

    if(sample.total_weight > 0.0) {
        for(cd& coeff : sample.coeffs)
            coeff /= sample.total_weight;
    }

    return sample;
}

ProductGSample BootstrapProductSampleFromSubsamples(
    const std::vector<std::vector<cd>>& subsample_coeff_sums,
    const std::vector<double>& subsample_counts,
    TRandom3& rng
)
{
    ProductGSample sample;
    if(subsample_coeff_sums.empty())
        return sample;

    sample.coeffs.assign(subsample_coeff_sums.front().size(), cd(0.0, 0.0));
    std::vector<cd> compensation(sample.coeffs.size(), cd(0.0, 0.0));
    const int Nsub = static_cast<int>(subsample_coeff_sums.size());

    for(int isub = 0; isub < Nsub; ++isub) {
        const int pick = rng.Integer(Nsub);
        const auto& picked_coeffs =
            subsample_coeff_sums[static_cast<std::size_t>(pick)];

        AddProductCoefficientsCompensated(
            sample.coeffs,
            compensation,
            picked_coeffs
        );

        sample.total_weight +=
            subsample_counts[static_cast<std::size_t>(pick)];
    }

    if(sample.total_weight > 0.0) {
        for(cd& coeff : sample.coeffs)
            coeff /= sample.total_weight;
    }

    return sample;
}

MCGValue EvaluateProductGeneratingFunction(
    cd z,
    const ProductGSample& sample
)
{
    MCGValue value;
    if(sample.coeffs.empty() || sample.total_weight <= 0.0)
        return value;

    cd polynomial = sample.coeffs.back();
    cd derivative = cd(0.0, 0.0);

    for(std::size_t idx = sample.coeffs.size() - 1; idx > 0; --idx) {
        derivative = derivative * z + polynomial;
        polynomial = polynomial * z + sample.coeffs[idx - 1];
    }

    value.G = polynomial;
    value.dGdk = derivative;
    return value;
}

class ProductGCachedEvaluator {
public:
    explicit ProductGCachedEvaluator(const ProductGSample& sample)
        : sample_(sample) {}

    const MCGValue& Eval(const double* x) const {
        if(!cache_valid_ || x[0] != last_re_ || x[1] != last_im_) {
            last_re_ = x[0];
            last_im_ = x[1];
            cached_value_ =
                EvaluateProductGeneratingFunction(cd(last_re_, last_im_), sample_);
            cache_valid_ = true;
        }

        return cached_value_;
    }

private:
    const ProductGSample& sample_;
    mutable bool cache_valid_ = false;
    mutable double last_re_ = 0.0;
    mutable double last_im_ = 0.0;
    mutable MCGValue cached_value_;
};

class ReProductGGrad : public ROOT::Math::IGradientFunctionMultiDim {
public:
    explicit ReProductGGrad(const ProductGCachedEvaluator& evaluator)
        : evaluator_(&evaluator) {}

    unsigned int NDim() const override { return 2; }

    ROOT::Math::IBaseFunctionMultiDim* Clone() const override {
        return new ReProductGGrad(*this);
    }

private:
    double DoEval(const double* x) const override {
        return std::real(evaluator_->Eval(x).G);
    }

    double DoDerivative(const double* x, unsigned int icoord) const override {
        cd dGdz = evaluator_->Eval(x).dGdk;
        return (icoord == 0) ? std::real(dGdz) : -std::imag(dGdz);
    }

    const ProductGCachedEvaluator* evaluator_;
};

class ImProductGGrad : public ROOT::Math::IGradientFunctionMultiDim {
public:
    explicit ImProductGGrad(const ProductGCachedEvaluator& evaluator)
        : evaluator_(&evaluator) {}

    unsigned int NDim() const override { return 2; }

    ROOT::Math::IBaseFunctionMultiDim* Clone() const override {
        return new ImProductGGrad(*this);
    }

private:
    double DoEval(const double* x) const override {
        return std::imag(evaluator_->Eval(x).G);
    }

    double DoDerivative(const double* x, unsigned int icoord) const override {
        cd dGdz = evaluator_->Eval(x).dGdk;
        return (icoord == 0) ? std::imag(dGdz) : std::real(dGdz);
    }

    const ProductGCachedEvaluator* evaluator_;
};

RootResult FindProductGeneratingZero(
    const ProductGSample& sample,
    double x0_re,
    double x0_im
)
{
    RootResult result;

    try {
        auto t1 = std::chrono::high_resolution_clock::now();

        ROOT::Math::GSLMultiRootFinder finder(
            ROOT::Math::GSLMultiRootFinder::kHybridSJ
        );

        ProductGCachedEvaluator evaluator(sample);
        ReProductGGrad f_re(evaluator);
        ImProductGGrad f_im(evaluator);
        finder.AddFunction(f_re);
        finder.AddFunction(f_im);

        double x0[2] = {x0_re, x0_im};
        bool ok = finder.Solve(x0, 1000, 1e-10, 1e-10);

        const double* root = finder.X();
        const double* fval = finder.FVal();

        result.ok = ok;
        result.status = finder.Status();
        if(root && fval) {
            result.k = {root[0], root[1]};
            result.F = {fval[0], fval[1]};
        }

        auto t2 = std::chrono::high_resolution_clock::now();
        result.time_ms =
            std::chrono::duration<double, std::milli>(t2 - t1).count();
    }
    catch(const std::exception&) {
        result.ok = false;
        result.status = -999;
    }

    return result;
}

std::vector<RootResult> FindProductGeneratingZerosFromStarts(
    const ProductGSample& sample,
    const std::vector<std::pair<double, double>>& start_points,
    int ires = -1,
    std::vector<RootDiagnostic>* diagnostics = nullptr
)
{
    std::vector<RootResult> roots;

    double max_root_size = 0.0;
    for(const auto& p : start_points) {
        max_root_size = std::max(
            max_root_size,
            std::sqrt(p.first * p.first + p.second * p.second)
        );
    }
    max_root_size *= 2.0;
    if(max_root_size <= 0.0)
        max_root_size = std::numeric_limits<double>::infinity();

    for(const auto& p : start_points) {
        RootResult r = FindProductGeneratingZero(sample, p.first, p.second);
        if(diagnostics) {
            const cd z_start(p.first, p.second);
            const MCGValue start_value =
                EvaluateProductGeneratingFunction(z_start, sample);

            cd z_linearized(NAN, NAN);
            if(std::abs(start_value.dGdk) > 0.0 &&
               std::isfinite(start_value.dGdk.real()) &&
               std::isfinite(start_value.dGdk.imag())) {
                z_linearized = z_start - start_value.G / start_value.dGdk;
            }

            RootDiagnostic diagnostic;
            diagnostic.ires = ires;
            diagnostic.iroot =
                static_cast<int>(diagnostics->size());
            diagnostic.z_start = z_start;
            diagnostic.G_start = start_value.G;
            diagnostic.Gprime_start = start_value.dGdk;
            diagnostic.z_linearized = z_linearized;
            diagnostic.tracked = r;
            diagnostics->push_back(diagnostic);
        }
        if(IsGoodMCGRoot(r, max_root_size))
            roots.push_back(r);
    }

    const double duplicate_tol = 1e-6;
    std::vector<RootResult> unique_roots;

    for(const auto& r : roots) {
        bool duplicate = false;
        for(const auto& u : unique_roots) {
            if(std::abs(r.k - u.k) < duplicate_tol) {
                duplicate = true;
                break;
            }
        }

        if(!duplicate)
            unique_roots.push_back(r);
    }

    std::sort(
        unique_roots.begin(),
        unique_roots.end(),
        [](const RootResult& a, const RootResult& b) {
            return std::abs(a.k) < std::abs(b.k);
        }
    );

    return unique_roots;
}

void WriteMCGeneratingRoots(
    const std::string& filename,
    const std::vector<RootResult>& roots
)
{
    std::ofstream out(filename);
    out << std::setprecision(17);
    out << "# iroot  Re(k)  Im(k)  Re(G)  Im(G)  time_ms\n";

    for(std::size_t i = 0; i < roots.size(); ++i) {
        const auto& r = roots[i];
        out << i << " "
            << r.k.real() << " "
            << r.k.imag() << " "
            << r.F.real() << " "
            << r.F.imag() << " "
            << r.time_ms << "\n";
    }
}

void WriteBootstrapMCGeneratingRoots(
    const std::string& filename,
    const std::vector<std::vector<RootResult>>& roots_from_resampling
)
{
    std::ofstream out(filename);
    out << std::setprecision(17);
    out << "# ires  iroot  ok  status  Re(k)  Im(k)  Re(G)  Im(G)  time_ms\n";

    for(std::size_t ires = 0; ires < roots_from_resampling.size(); ++ires) {
        const auto& roots = roots_from_resampling[ires];

        for(std::size_t iroot = 0; iroot < roots.size(); ++iroot) {
            const auto& r = roots[iroot];
            out << ires << " "
                << iroot << " "
                << (r.ok ? 1 : 0) << " "
                << r.status << " "
                << r.k.real() << " "
                << r.k.imag() << " "
                << r.F.real() << " "
                << r.F.imag() << " "
                << r.time_ms << "\n";
        }
    }
}

void WriteRootDiagnostics(
    const std::string& filename,
    const std::vector<RootDiagnostic>& diagnostics
)
{
    if(filename.empty())
        return;

    std::ofstream out(filename);
    if(!out.is_open())
        throw std::runtime_error("Cannot open " + filename);

    out << std::setprecision(17);
    out << "# ires iroot "
        << "Re(z_start) Im(z_start) "
        << "Re(G_start) Im(G_start) "
        << "Re(Gprime_start) Im(Gprime_start) "
        << "Re(z_linearized) Im(z_linearized) "
        << "Re(z_tracked) Im(z_tracked) "
        << "Re(G_tracked) Im(G_tracked) "
        << "ok status time_ms\n";

    for(const RootDiagnostic& diagnostic : diagnostics) {
        out << diagnostic.ires << " "
            << diagnostic.iroot << " "
            << diagnostic.z_start.real() << " "
            << diagnostic.z_start.imag() << " "
            << diagnostic.G_start.real() << " "
            << diagnostic.G_start.imag() << " "
            << diagnostic.Gprime_start.real() << " "
            << diagnostic.Gprime_start.imag() << " "
            << diagnostic.z_linearized.real() << " "
            << diagnostic.z_linearized.imag() << " "
            << diagnostic.tracked.k.real() << " "
            << diagnostic.tracked.k.imag() << " "
            << diagnostic.tracked.F.real() << " "
            << diagnostic.tracked.F.imag() << " "
            << (diagnostic.tracked.ok ? 1 : 0) << " "
            << diagnostic.tracked.status << " "
            << diagnostic.tracked.time_ms << "\n";
    }
}

void WriteProductCoefficientComparison(
    const std::string& filename,
    const ProductGSample& before,
    const ProductGSample& after,
    const std::vector<double>& coefficient_factors
)
{
    std::ofstream out(filename);
    if(!out.is_open())
        throw std::runtime_error("Cannot open " + filename);

    out << std::setprecision(17);
    out << "# order factor Re(a_before) Im(a_before) Re(a_after) Im(a_after)\n";
    for(std::size_t order = 0; order < before.coeffs.size(); ++order) {
        out << order << " "
            << coefficient_factors[order] << " "
            << before.coeffs[order].real() << " "
            << before.coeffs[order].imag() << " "
            << after.coeffs[order].real() << " "
            << after.coeffs[order].imag() << "\n";
    }
}

void WriteBootstrapProductCoefficientComparison(
    const std::string& filename,
    const std::vector<ProductGSample>& before,
    const std::vector<ProductGSample>& after,
    const std::vector<double>& coefficient_factors
)
{
    std::ofstream out(filename);
    if(!out.is_open())
        throw std::runtime_error("Cannot open " + filename);

    out << std::setprecision(17);
    out << "# ires order factor Re(a_before) Im(a_before) Re(a_after) Im(a_after)\n";
    for(std::size_t ires = 0; ires < before.size(); ++ires) {
        for(std::size_t order = 0; order < before[ires].coeffs.size(); ++order) {
            out << ires << " "
                << order << " "
                << coefficient_factors[order] << " "
                << before[ires].coeffs[order].real() << " "
                << before[ires].coeffs[order].imag() << " "
                << after[ires].coeffs[order].real() << " "
                << after[ires].coeffs[order].imag() << "\n";
        }
    }
}

void WriteProductCoefficients(
    const std::string& filename,
    const ProductGSample& sample
)
{
    std::ofstream out(filename);
    if(!out.is_open())
        throw std::runtime_error("Cannot open " + filename);

    out << std::setprecision(17);
    out << "# order Re(a) Im(a)\n";
    for(std::size_t order = 0; order < sample.coeffs.size(); ++order) {
        out << order << " "
            << sample.coeffs[order].real() << " "
            << sample.coeffs[order].imag() << "\n";
    }
}

void WriteBootstrapProductCoefficients(
    const std::string& filename,
    const std::vector<ProductGSample>& samples
)
{
    std::ofstream out(filename);
    if(!out.is_open())
        throw std::runtime_error("Cannot open " + filename);

    out << std::setprecision(17);
    out << "# ires order Re(a) Im(a)\n";
    for(std::size_t ires = 0; ires < samples.size(); ++ires) {
        for(std::size_t order = 0; order < samples[ires].coeffs.size(); ++order) {
            out << ires << " "
                << order << " "
                << samples[ires].coeffs[order].real() << " "
                << samples[ires].coeffs[order].imag() << "\n";
        }
    }
}

double CoefficientChi2AtPoint(
    cd z,
    const ProductGSample& nominal_sample,
    const std::vector<std::size_t>& orders,
    const std::vector<std::vector<double>>& covariance,
    double* sigma_trace = nullptr,
    cd* polynomial_value = nullptr
)
{
    const std::size_t ncoeff = orders.size();
    if(ncoeff == 0 || covariance.size() != ncoeff)
        return NAN;
    for(std::size_t i = 0; i < ncoeff; ++i) {
        if(covariance[i].size() != ncoeff)
            return NAN;
    }

    cd polynomial(0.0, 0.0);
    std::vector<cd> basis(ncoeff, cd(1.0, 0.0));
    for(std::size_t i = 0; i < ncoeff; ++i) {
        basis[i] = std::pow(z, static_cast<int>(orders[i]));
        polynomial += nominal_sample.coeffs[orders[i]] * basis[i];
    }

    double sigma_re_re = 0.0;
    double sigma_re_im = 0.0;
    double sigma_im_im = 0.0;
    for(std::size_t i = 0; i < ncoeff; ++i) {
        const double basis_i_re = basis[i].real();
        const double basis_i_im = basis[i].imag();
        for(std::size_t j = 0; j < ncoeff; ++j) {
            const double cov = covariance[i][j];
            sigma_re_re += basis_i_re * cov * basis[j].real();
            sigma_re_im += basis_i_re * cov * basis[j].imag();
            sigma_im_im += basis_i_im * cov * basis[j].imag();
        }
    }

    double trace = sigma_re_re + sigma_im_im;
    const double regularization =
        std::max(1e-30, 1e-14 * std::max(1.0, std::abs(trace)));
    sigma_re_re += regularization;
    sigma_im_im += regularization;
    trace = sigma_re_re + sigma_im_im;

    const double determinant =
        sigma_re_re * sigma_im_im - sigma_re_im * sigma_re_im;
    if(determinant <= 0.0 || !std::isfinite(determinant))
        return NAN;

    const double p_re = polynomial.real();
    const double p_im = polynomial.imag();
    const double chi2 =
        (
            sigma_im_im * p_re * p_re -
            2.0 * sigma_re_im * p_re * p_im +
            sigma_re_re * p_im * p_im
        ) /
        determinant;

    if(sigma_trace)
        *sigma_trace = trace;
    if(polynomial_value)
        *polynomial_value = polynomial;

    return chi2;
}

std::vector<std::vector<double>> BuildRealCoefficientCovariance(
    const std::vector<ProductGSample>& bootstrap_samples,
    const std::vector<std::size_t>& orders
)
{
    const std::size_t ncoeff = orders.size();
    const std::size_t nsample = bootstrap_samples.size();
    std::vector<std::vector<double>> covariance(
        ncoeff,
        std::vector<double>(ncoeff, 0.0)
    );
    if(ncoeff == 0 || nsample < 2)
        return covariance;

    std::vector<double> mean(ncoeff, 0.0);
    for(const ProductGSample& sample : bootstrap_samples) {
        for(std::size_t i = 0; i < ncoeff; ++i)
            mean[i] += sample.coeffs[orders[i]].real();
    }
    for(double& value : mean)
        value /= static_cast<double>(nsample);

    for(const ProductGSample& sample : bootstrap_samples) {
        for(std::size_t i = 0; i < ncoeff; ++i) {
            const double delta_i = sample.coeffs[orders[i]].real() - mean[i];
            for(std::size_t j = 0; j < ncoeff; ++j) {
                const double delta_j =
                    sample.coeffs[orders[j]].real() - mean[j];
                covariance[i][j] += delta_i * delta_j;
            }
        }
    }

    const double inverse_dof = 1.0 / static_cast<double>(nsample - 1);
    for(std::size_t i = 0; i < ncoeff; ++i) {
        for(std::size_t j = 0; j < ncoeff; ++j)
            covariance[i][j] *= inverse_dof;
    }

    return covariance;
}

void WriteCoefficientChi2Scan(
    const std::string& filename,
    const ProductGSample& nominal_sample,
    const std::vector<ProductGSample>& bootstrap_samples,
    const std::vector<RootResult>& nominal_roots,
    const std::vector<std::pair<double, double>>& root_start_points,
    bool remove_odd_coefficients,
    int grid_points,
    double requested_half_width,
    double re_min,
    double re_max,
    double im_min,
    double im_max
)
{
    if(filename.empty())
        return;
    if(bootstrap_samples.size() < 2)
        return;
    if(nominal_sample.coeffs.empty())
        return;

    const bool use_explicit_bounds =
        std::isfinite(re_min) &&
        std::isfinite(re_max) &&
        std::isfinite(im_min) &&
        std::isfinite(im_max);

    cd center(0.0, 0.0);
    double scan_re_min = re_min;
    double scan_re_max = re_max;
    double scan_im_min = im_min;
    double scan_im_max = im_max;
    double half_width = NAN;

    if(!use_explicit_bounds) {
        if(!nominal_roots.empty()) {
            center = nominal_roots.front().k;
        } else if(!root_start_points.empty()) {
            center = cd(root_start_points.front().first, root_start_points.front().second);
        }

        half_width =
            requested_half_width > 0.0
                ? requested_half_width
                : 0.25 * std::max(1.0, std::abs(center));
        scan_re_min = center.real() - half_width;
        scan_re_max = center.real() + half_width;
        scan_im_min = center.imag() - half_width;
        scan_im_max = center.imag() + half_width;
    } else {
        center = cd(0.5 * (scan_re_min + scan_re_max),
                    0.5 * (scan_im_min + scan_im_max));
    }

    std::vector<std::size_t> orders;
    for(std::size_t order = 0; order < nominal_sample.coeffs.size(); ++order) {
        if(remove_odd_coefficients && order % 2 == 1)
            continue;
        orders.push_back(order);
    }
    const std::vector<std::vector<double>> covariance =
        BuildRealCoefficientCovariance(bootstrap_samples, orders);

    struct GridRow {
        double re = 0.0;
        double im = 0.0;
        double chi2 = NAN;
        double sigma_trace = NAN;
        cd polynomial = {NAN, NAN};
    };

    std::vector<GridRow> rows;
    rows.reserve(static_cast<std::size_t>(grid_points * grid_points));
    double chi2_min = std::numeric_limits<double>::infinity();

    for(int ix = 0; ix < grid_points; ++ix) {
        const double re =
            scan_re_min +
            (scan_re_max - scan_re_min) * static_cast<double>(ix) /
                static_cast<double>(grid_points - 1);
        for(int iy = 0; iy < grid_points; ++iy) {
            const double im =
                scan_im_min +
                (scan_im_max - scan_im_min) * static_cast<double>(iy) /
                    static_cast<double>(grid_points - 1);

            GridRow row;
            row.re = re;
            row.im = im;
            row.chi2 = CoefficientChi2AtPoint(
                cd(re, im),
                nominal_sample,
                orders,
                covariance,
                &row.sigma_trace,
                &row.polynomial
            );
            if(std::isfinite(row.chi2))
                chi2_min = std::min(chi2_min, row.chi2);
            rows.push_back(row);
        }
    }

    std::ofstream out(filename);
    if(!out.is_open())
        throw std::runtime_error("Cannot open " + filename);

    out << std::setprecision(17);
    out << "# coefficient covariance chi2 scan\n"
        << "# mode " << (use_explicit_bounds ? "global" : "local") << "\n"
        << "# center_Re " << center.real()
        << " center_Im " << center.imag()
        << " half_width " << half_width
        << " re_min " << scan_re_min
        << " re_max " << scan_re_max
        << " im_min " << scan_im_min
        << " im_max " << scan_im_max
        << " grid_points " << grid_points
        << " chi2_min " << chi2_min << "\n"
        << "# contours: delta_chi2 1sigma=2.30 2sigma=6.18 3sigma=11.83\n"
        << "# Re(z) Im(z) chi2 delta_chi2 sigma_trace Re(P) Im(P)\n";

    for(const GridRow& row : rows) {
        out << row.re << " "
            << row.im << " "
            << row.chi2 << " "
            << (row.chi2 - chi2_min) << " "
            << row.sigma_trace << " "
            << row.polynomial.real() << " "
            << row.polynomial.imag() << "\n";
    }
}

std::vector<std::vector<double>> BuildQSubsampleBins(
    const std::vector<double>& Q,
    int Nsub,
    const QBinLayout& layout
)
{
    std::vector<std::vector<double>> subsamples(
        static_cast<std::size_t>(Nsub),
        std::vector<double>(static_cast<std::size_t>(layout.bins), 0.0)
    );

    for(std::size_t i = 0; i < Q.size(); ++i) {
        const std::size_t isub = i % static_cast<std::size_t>(Nsub);
        const std::size_t ibin =
            static_cast<std::size_t>(QBinIndex(Q[i], layout));
        subsamples[isub][ibin] += 1.0;
    }

    return subsamples;
}

MCGSample BootstrapQSampleFromSubsampleBins(
    const std::vector<std::vector<double>>& subsample_bins,
    const QBinLayout& layout,
    TRandom3& rng
)
{
    std::vector<double> bin_weights(static_cast<std::size_t>(layout.bins), 0.0);

    const int Nsub = static_cast<int>(subsample_bins.size());

    for(int isub = 0; isub < Nsub; ++isub) {
        const int pick = rng.Integer(Nsub);
        const auto& picked_bins = subsample_bins[static_cast<std::size_t>(pick)];

        for(int ibin = 0; ibin < layout.bins; ++ibin)
            bin_weights[static_cast<std::size_t>(ibin)] +=
                picked_bins[static_cast<std::size_t>(ibin)];
    }

    MCGSample sample;
    sample.q.reserve(static_cast<std::size_t>(layout.bins));
    sample.weight.reserve(static_cast<std::size_t>(layout.bins));

    for(int ibin = 0; ibin < layout.bins; ++ibin) {
        const double weight = bin_weights[static_cast<std::size_t>(ibin)];
        if(weight == 0.0) continue;

        const double q_center =
            layout.q_min + (static_cast<double>(ibin) + 0.5) * layout.width;

        sample.q.push_back(q_center);
        sample.weight.push_back(weight);
        sample.total_weight += weight;
    }

    return sample;
}

MCGSample BuildQSampleFromSubsampleBins(
    const std::vector<std::vector<double>>& subsample_bins,
    const QBinLayout& layout
)
{
    std::vector<double> bin_weights(static_cast<std::size_t>(layout.bins), 0.0);

    for(const auto& sub : subsample_bins) {
        for(int ibin = 0; ibin < layout.bins; ++ibin)
            bin_weights[static_cast<std::size_t>(ibin)] +=
                sub[static_cast<std::size_t>(ibin)];
    }

    MCGSample sample;
    sample.q.reserve(static_cast<std::size_t>(layout.bins));
    sample.weight.reserve(static_cast<std::size_t>(layout.bins));

    for(int ibin = 0; ibin < layout.bins; ++ibin) {
        const double weight = bin_weights[static_cast<std::size_t>(ibin)];
        if(weight == 0.0) continue;

        const double q_center =
            layout.q_min + (static_cast<double>(ibin) + 0.5) * layout.width;

        sample.q.push_back(q_center);
        sample.weight.push_back(weight);
        sample.total_weight += weight;
    }

    return sample;
}

std::vector<std::vector<double>> BuildExactQSubsamples(
    const std::vector<double>& Q,
    int Nsub
)
{
    std::vector<std::vector<double>> subsamples(static_cast<std::size_t>(Nsub));

    for(std::size_t i = 0; i < Q.size(); ++i) {
        const std::size_t isub = i % static_cast<std::size_t>(Nsub);
        subsamples[isub].push_back(Q[i]);
    }

    return subsamples;
}

MCGSample BootstrapExactQSampleFromSubsamples(
    const std::vector<std::vector<double>>& subsamples,
    TRandom3& rng
)
{
    std::size_t total_size = 0;
    for(const auto& sub : subsamples)
        total_size += sub.size();

    MCGSample sample;
    sample.q.reserve(total_size);
    sample.weight.reserve(total_size);

    const int Nsub = static_cast<int>(subsamples.size());

    for(int isub = 0; isub < Nsub; ++isub) {
        const int pick = rng.Integer(Nsub);
        const auto& picked_subsample = subsamples[static_cast<std::size_t>(pick)];

        sample.q.insert(
            sample.q.end(),
            picked_subsample.begin(),
            picked_subsample.end()
        );
    }

    sample.weight.assign(sample.q.size(), 1.0);
    sample.total_weight = static_cast<double>(sample.q.size());
    return sample;
}

// ------------------------------------------------------------
// Probability density p(phi)
// ------------------------------------------------------------
double p_phi(double phi, double v2, double psi2)
{
    return (1.0 + 2.0 * v2 * std::cos(2.0 * (phi - psi2))) / (2.0 * M_PI);
}

// ------------------------------------------------------------
// Generate one event with M particles using accept/reject
// ------------------------------------------------------------
struct ToyEvent {
    std::vector<double> phi;
    std::vector<double> eta;
};

double SampleEta(double eta_max, TRandom3& rng)
{
    return rng.Uniform(-eta_max, eta_max);
}

double SampleSmearedEta(double eta_center, double delta_eta, double eta_max, TRandom3& rng)
{
    if(delta_eta <= 0.0)
        return eta_center;

    for(int itry = 0; itry < 100; ++itry) {
        const double eta = eta_center + rng.Uniform(-delta_eta, delta_eta);
        if(eta >= -eta_max && eta <= eta_max)
            return eta;
    }

    const double eta = eta_center + rng.Uniform(-delta_eta, delta_eta);
    return std::max(-eta_max, std::min(eta_max, eta));
}

ToyEvent generate_event(double v2, double psi2, int M, double eta_max, TRandom3& rng)
{
    const double p_max = (1.0 + 2.0 * std::abs(v2)) / (2.0 * M_PI);

    ToyEvent event;
    event.phi.resize(static_cast<std::size_t>(M));
    event.eta.resize(static_cast<std::size_t>(M));
    int naccepted = 0;

    while(naccepted < M)
    {
        double phi = rng.Uniform(0.0, 2.0 * M_PI);
        double y   = rng.Uniform(0.0, p_max);

        double p = p_phi(phi, v2, psi2);

        if(y < p)
        {
            const std::size_t index = static_cast<std::size_t>(naccepted);
            event.phi[index] = phi;
            event.eta[index] = SampleEta(eta_max, rng);
            ++naccepted;
        }
    }

    return event;
}

double WrapPhi(double phi)
{
    phi = std::fmod(phi, 2.0 * M_PI);
    if(phi < 0.0)
        phi += 2.0 * M_PI;
    return phi;
}

void AppendCopiedUniformNonflow(
    ToyEvent& event,
    int nonflow_groups,
    int nonflow_q,
    double nonflow_delta,
    double nonflow_delta_eta,
    double eta_max,
    TRandom3& rng
)
{
    if(nonflow_q <= 0 || nonflow_groups <= 0)
        return;

    event.phi.reserve(
        event.phi.size() +
        static_cast<std::size_t>(nonflow_groups) *
            static_cast<std::size_t>(nonflow_q)
    );
    event.eta.reserve(event.phi.capacity());

    for(int igroup = 0; igroup < nonflow_groups; ++igroup) {
        const double phi = rng.Uniform(0.0, 2.0 * M_PI);
        const double eta = SampleEta(eta_max, rng);

        for(int icopy = 0; icopy < nonflow_q; ++icopy) {
            const double smeared_phi =
                nonflow_delta > 0.0
                    ? WrapPhi(phi + rng.Uniform(-nonflow_delta, nonflow_delta))
                    : phi;
            const double smeared_eta =
                SampleSmearedEta(eta, nonflow_delta_eta, eta_max, rng);
            event.phi.push_back(smeared_phi);
            event.eta.push_back(smeared_eta);
        }
    }
}

ToyEvent ConcatenateEvents(const ToyEvent& first, const ToyEvent& second)
{
    ToyEvent combined;
    combined.phi.reserve(first.phi.size() + second.phi.size());
    combined.eta.reserve(first.eta.size() + second.eta.size());
    combined.phi.insert(combined.phi.end(), first.phi.begin(), first.phi.end());
    combined.phi.insert(combined.phi.end(), second.phi.begin(), second.phi.end());
    combined.eta.insert(combined.eta.end(), first.eta.begin(), first.eta.end());
    combined.eta.insert(combined.eta.end(), second.eta.begin(), second.eta.end());
    return combined;
}

int EtaBinIndex(double eta, double eta_max, int Neta)
{
    const double eta_width = 2.0 * eta_max / static_cast<double>(Neta);
    int ibin = static_cast<int>((eta + eta_max) / eta_width);
    if(ibin < 0)
        ibin = 0;
    if(ibin >= Neta)
        ibin = Neta - 1;
    return ibin;
}

std::vector<cd> BuildEtaSubeventCrossProductCoefficients(
    int n,
    double theta,
    const ToyEvent& event,
    double eta_max,
    int Neta
)
{
    std::vector<cd> eta_bin_sums(static_cast<std::size_t>(Neta), cd(0.0, 0.0));
    std::vector<double> eta_bin_counts(static_cast<std::size_t>(Neta), 0.0);
    for(std::size_t iparticle = 0; iparticle < event.phi.size(); ++iparticle) {
        const int ibin = EtaBinIndex(event.eta[iparticle], eta_max, Neta);
        eta_bin_sums[static_cast<std::size_t>(ibin)] +=
            cd(0.0, std::cos(n * (event.phi[iparticle] - theta)));
        eta_bin_counts[static_cast<std::size_t>(ibin)] += 1.0;
    }

    const std::vector<cd> ordinary_coeffs =
        BuildProductCoefficients(n, theta, event.phi);
    const std::vector<double> ordinary_factors =
        BuildExactProductFactors(static_cast<int>(event.phi.size()));
    std::vector<cd> coeffs(event.phi.size() + 1, cd(0.0, 0.0));
    coeffs[0] = cd(1.0, 0.0);
    const double multiplicity = static_cast<double>(event.phi.size());
    double multiplicity_power = 1.0;
    double factorial = 1.0;
    for(std::size_t order = 1; order < coeffs.size(); ++order) {
        multiplicity_power *= multiplicity;
        factorial *= static_cast<double>(order);
        if(order > static_cast<std::size_t>(Neta)) {
            coeffs[order] = ordinary_coeffs[order] * ordinary_factors[order];
            continue;
        }

        const int groups = static_cast<int>(order);
        cd coefficient_product(1.0, 0.0);
        double allowed_count_product = 1.0;
        for(int igroup = 0; igroup < groups; ++igroup) {
            const int begin_bin = igroup * Neta / groups;
            const int end_bin = (igroup + 1) * Neta / groups;
            cd group_sum(0.0, 0.0);
            double group_count = 0.0;
            for(int ibin = begin_bin; ibin < end_bin; ++ibin) {
                group_sum += eta_bin_sums[static_cast<std::size_t>(ibin)];
                group_count += eta_bin_counts[static_cast<std::size_t>(ibin)];
            }
            coefficient_product *= group_sum;
            allowed_count_product *= group_count;
        }

        coeffs[order] = coefficient_product;
        if(allowed_count_product > 0.0) {
            coeffs[order] *=
                multiplicity_power /
                (factorial * allowed_count_product);
        } else {
            coeffs[order] = cd(0.0, 0.0);
        }
    }

    return coeffs;
}

std::vector<cd> BuildThetaAveragedEtaSubeventProductCoefficients(
    int n,
    double theta_offset,
    int theta_bins,
    const ToyEvent& event,
    double eta_max,
    int Neta
)
{
    std::vector<cd> average(event.phi.size() + 1, cd(0.0, 0.0));
    std::vector<cd> compensation(event.phi.size() + 1, cd(0.0, 0.0));
    const double theta_step =
        2.0 * M_PI / (static_cast<double>(n) * theta_bins);

    for(int itheta = 0; itheta < theta_bins; ++itheta) {
        const std::vector<cd> coefficients =
            BuildEtaSubeventCrossProductCoefficients(
                n,
                theta_offset + itheta * theta_step,
                event,
                eta_max,
                Neta
            );
        for(std::size_t order = 0; order < average.size(); ++order) {
            const cd corrected = coefficients[order] - compensation[order];
            const cd updated = average[order] + corrected;
            compensation[order] = (updated - average[order]) - corrected;
            average[order] = updated;
        }
    }

    const double inverse_bins = 1.0 / theta_bins;
    for(cd& coefficient : average)
        coefficient *= inverse_bins;

    return average;
}

// ------------------------------------------------------------
// Triangular distribution: min=0, mode=a, max=b
// Equivalent to numpy.random.triangular(0,a,b)
// ------------------------------------------------------------
double sample_v2(double a, double b, TRandom3& rng)
{
    double u = rng.Uniform(0.0, 1.0);
    double c = a / b;

    if(u < c)
        return std::sqrt(u * a * b);
    else
        return b - std::sqrt((1.0 - u) * b * (b - a));
}

// ------------------------------------------------------------
// Complex Bessel J0 series
// Valid and fast enough for the range in this problem.
// ------------------------------------------------------------
cd besselJ0_series(cd z, double tol = 1e-15, int max_iter = 10000)
{
    cd zz = z * z;
    cd term = 1.0;
    cd sum  = term;

    for(int m = 1; m < max_iter; ++m)
    {
        term *= -zz / (4.0 * m * m);
        sum += term;

        if(std::abs(term) < tol * std::max(1.0, std::abs(sum)))
            break;
    }

    return sum;
}

// ------------------------------------------------------------
// Complex Bessel J1 series
// ------------------------------------------------------------
cd besselJ1_series(cd z, double tol = 1e-15, int max_iter = 10000)
{
    cd zz = z * z;
    cd term = 0.5 * z;
    cd sum  = term;

    for(int m = 1; m < max_iter; ++m)
    {
        term *= -zz / (4.0 * m * (m + 1.0));
        sum += term;

        if(std::abs(term) < tol * std::max(1.0, std::abs(sum)))
            break;
    }

    return sum;
}

struct DirectVnSample {
    std::vector<double> values;
    std::vector<double> weights;
    double total_weight = 0.0;
};

DirectVnSample BuildDirectVnSampleFromSubsamples(
    const std::vector<std::vector<double>>& subsample_counts,
    double vn_max)
{
    DirectVnSample sample;
    if(subsample_counts.empty())
        return sample;

    const std::size_t nbins = subsample_counts.front().size();
    sample.values.resize(nbins);
    sample.weights.assign(nbins, 0.0);
    const double width = vn_max / static_cast<double>(nbins);

    for(std::size_t ibin = 0; ibin < nbins; ++ibin)
        sample.values[ibin] = (static_cast<double>(ibin) + 0.5) * width;

    for(const auto& subsample : subsample_counts) {
        for(std::size_t ibin = 0; ibin < nbins; ++ibin) {
            sample.weights[ibin] += subsample[ibin];
            sample.total_weight += subsample[ibin];
        }
    }

    if(sample.total_weight > 0.0) {
        for(double& weight : sample.weights)
            weight /= sample.total_weight;
    }

    return sample;
}

DirectVnSample BuildDirectVnSampleFromCounts(
    const std::vector<double>& counts,
    double vn_max)
{
    DirectVnSample sample;
    const std::size_t nbins = counts.size();
    sample.values.resize(nbins);
    sample.weights.assign(nbins, 0.0);
    const double width = vn_max / static_cast<double>(nbins);

    for(std::size_t ibin = 0; ibin < nbins; ++ibin) {
        sample.values[ibin] = (static_cast<double>(ibin) + 0.5) * width;
        sample.weights[ibin] = counts[ibin];
        sample.total_weight += counts[ibin];
    }

    if(sample.total_weight > 0.0) {
        for(double& weight : sample.weights)
            weight /= sample.total_weight;
    }

    return sample;
}

std::vector<DirectVnSample> BuildDirectVnSamplesForSubsamples(
    const std::vector<std::vector<double>>& subsample_counts,
    double vn_max)
{
    std::vector<DirectVnSample> samples;
    samples.reserve(subsample_counts.size());
    for(const auto& counts : subsample_counts)
        samples.push_back(BuildDirectVnSampleFromCounts(counts, vn_max));
    return samples;
}

DirectVnSample BootstrapDirectVnSampleFromSubsamples(
    const std::vector<std::vector<double>>& subsample_counts,
    double vn_max,
    TRandom3& rng)
{
    DirectVnSample sample;
    if(subsample_counts.empty())
        return sample;

    const int nsub = static_cast<int>(subsample_counts.size());
    const std::size_t nbins = subsample_counts.front().size();
    sample.values.resize(nbins);
    sample.weights.assign(nbins, 0.0);
    const double width = vn_max / static_cast<double>(nbins);

    for(std::size_t ibin = 0; ibin < nbins; ++ibin)
        sample.values[ibin] = (static_cast<double>(ibin) + 0.5) * width;

    for(int isub = 0; isub < nsub; ++isub) {
        const int pick = rng.Integer(nsub);
        const auto& picked = subsample_counts[static_cast<std::size_t>(pick)];
        for(std::size_t ibin = 0; ibin < nbins; ++ibin) {
            sample.weights[ibin] += picked[ibin];
            sample.total_weight += picked[ibin];
        }
    }

    if(sample.total_weight > 0.0) {
        for(double& weight : sample.weights)
            weight /= sample.total_weight;
    }

    return sample;
}

MCGValue EvaluateDirectVnLYZTruncated(
    cd z,
    const DirectVnSample& sample,
    double k_scale,
    int max_order)
{
    MCGValue value;
    if(sample.total_weight <= 0.0)
        return value;

    for(std::size_t ibin = 0; ibin < sample.values.size(); ++ibin) {
        const double weight = sample.weights[ibin];
        if(weight <= 0.0)
            continue;
        const double vn = sample.values[ibin];
        const cd x = k_scale * vn * z;
        const cd dx_dz(k_scale * vn, 0.0);

        cd term(1.0, 0.0);
        cd dterm(0.0, 0.0);
        cd sum = term;
        cd dsum = dterm;

        for(int m = 1; m <= max_order; ++m) {
            const double denom = 4.0 * static_cast<double>(m) * m;
            const cd ratio = -(x * x) / denom;
            const cd dratio = -(x * dx_dz) / (2.0 * static_cast<double>(m) * m);
            const cd previous_term = term;
            const cd previous_dterm = dterm;

            term = previous_term * ratio;
            dterm = previous_dterm * ratio + previous_term * dratio;
            sum += term;
            dsum += dterm;
        }

        value.G += weight * sum;
        value.dGdk += weight * dsum;
    }

    return value;
}

MCGValue EvaluateDirectVnJ0(cd k, const DirectVnSample& sample)
{
    MCGValue value;
    if(sample.total_weight <= 0.0)
        return value;

    for(std::size_t ibin = 0; ibin < sample.values.size(); ++ibin) {
        const double weight = sample.weights[ibin];
        if(weight <= 0.0)
            continue;
        const double vn = sample.values[ibin];
        value.G += weight * besselJ0_series(k * vn);
        value.dGdk += weight * (-vn) * besselJ1_series(k * vn);
    }

    return value;
}

MCGValue EvaluateDirectVnGeneratingFunction(
    cd z,
    const DirectVnSample& sample,
    double k_scale,
    int max_order)
{
    if(max_order >= 0)
        return EvaluateDirectVnLYZTruncated(z, sample, k_scale, max_order);
    return EvaluateDirectVnJ0(z, sample);
}

double DirectVnFunctionChi2AtPoint(
    cd z,
    const DirectVnSample& nominal_sample,
    const std::vector<DirectVnSample>& covariance_samples,
    bool covariance_samples_are_estimates,
    double k_scale,
    int max_order,
    double* sigma_trace = nullptr,
    cd* function_value = nullptr)
{
    if(covariance_samples.size() < 2)
        return NAN;

    const cd nominal =
        EvaluateDirectVnGeneratingFunction(z, nominal_sample, k_scale, max_order).G;

    double mean_re = 0.0;
    double mean_im = 0.0;
    std::vector<cd> values;
    values.reserve(covariance_samples.size());
    for(const DirectVnSample& sample : covariance_samples) {
        const cd value =
            EvaluateDirectVnGeneratingFunction(z, sample, k_scale, max_order).G;
        values.push_back(value);
        mean_re += value.real();
        mean_im += value.imag();
    }

    const double inverse_n = 1.0 / static_cast<double>(values.size());
    mean_re *= inverse_n;
    mean_im *= inverse_n;

    double sigma_re_re = 0.0;
    double sigma_re_im = 0.0;
    double sigma_im_im = 0.0;
    for(const cd value : values) {
        const double delta_re = value.real() - mean_re;
        const double delta_im = value.imag() - mean_im;
        sigma_re_re += delta_re * delta_re;
        sigma_re_im += delta_re * delta_im;
        sigma_im_im += delta_im * delta_im;
    }

    const double inverse_dof =
        1.0 / static_cast<double>(values.size() - 1);
    const double covariance_scale =
        covariance_samples_are_estimates
            ? inverse_dof
            : inverse_dof / static_cast<double>(values.size());
    sigma_re_re *= covariance_scale;
    sigma_re_im *= covariance_scale;
    sigma_im_im *= covariance_scale;

    double trace = sigma_re_re + sigma_im_im;
    const double regularization =
        std::max(1e-30, 1e-14 * std::max(1.0, std::abs(trace)));
    sigma_re_re += regularization;
    sigma_im_im += regularization;
    trace = sigma_re_re + sigma_im_im;

    const double determinant =
        sigma_re_re * sigma_im_im - sigma_re_im * sigma_re_im;
    if(determinant <= 0.0 || !std::isfinite(determinant))
        return NAN;

    const double g_re = nominal.real();
    const double g_im = nominal.imag();
    const double chi2 =
        (
            sigma_im_im * g_re * g_re -
            2.0 * sigma_re_im * g_re * g_im +
            sigma_re_re * g_im * g_im
        ) /
        determinant;

    if(sigma_trace)
        *sigma_trace = trace;
    if(function_value)
        *function_value = nominal;

    return chi2;
}

void WriteDirectVnFunctionChi2Scan(
    const std::string& filename,
    const std::string& label,
    const DirectVnSample& nominal_sample,
    const std::vector<DirectVnSample>& covariance_samples,
    bool covariance_samples_are_estimates,
    const std::vector<RootResult>& nominal_roots,
    const std::vector<std::pair<double, double>>& root_start_points,
    double k_scale,
    int max_order,
    int grid_points,
    double requested_half_width,
    double re_min,
    double re_max,
    double im_min,
    double im_max)
{
    if(filename.empty())
        return;
    if(covariance_samples.size() < 2)
        return;
    if(nominal_sample.total_weight <= 0.0)
        return;

    const bool use_explicit_bounds =
        std::isfinite(re_min) &&
        std::isfinite(re_max) &&
        std::isfinite(im_min) &&
        std::isfinite(im_max);

    cd center(0.0, 0.0);
    double scan_re_min = re_min;
    double scan_re_max = re_max;
    double scan_im_min = im_min;
    double scan_im_max = im_max;
    double half_width = NAN;

    if(!use_explicit_bounds) {
        if(!nominal_roots.empty()) {
            center = nominal_roots.front().k;
        } else if(!root_start_points.empty()) {
            center = cd(root_start_points.front().first, root_start_points.front().second);
        }

        half_width =
            requested_half_width > 0.0
                ? requested_half_width
                : 0.25 * std::max(1.0, std::abs(center));
        scan_re_min = center.real() - half_width;
        scan_re_max = center.real() + half_width;
        scan_im_min = center.imag() - half_width;
        scan_im_max = center.imag() + half_width;
    } else {
        center = cd(0.5 * (scan_re_min + scan_re_max),
                    0.5 * (scan_im_min + scan_im_max));
    }

    struct GridRow {
        double re = 0.0;
        double im = 0.0;
        double chi2 = NAN;
        double sigma_trace = NAN;
        cd value = {NAN, NAN};
    };

    std::vector<GridRow> rows(
        static_cast<std::size_t>(grid_points * grid_points)
    );
    double chi2_min = std::numeric_limits<double>::infinity();
    int completed_rows = 0;
    const int report_every = std::max(1, grid_points / 20);

    #pragma omp parallel for schedule(dynamic) reduction(min:chi2_min)
    for(int ix = 0; ix < grid_points; ++ix) {
        const double re =
            scan_re_min +
            (scan_re_max - scan_re_min) * static_cast<double>(ix) /
                static_cast<double>(grid_points - 1);
        for(int iy = 0; iy < grid_points; ++iy) {
            const double im =
                scan_im_min +
                (scan_im_max - scan_im_min) * static_cast<double>(iy) /
                    static_cast<double>(grid_points - 1);

            GridRow row;
            row.re = re;
            row.im = im;
            row.chi2 = DirectVnFunctionChi2AtPoint(
                cd(re, im),
                nominal_sample,
                covariance_samples,
                covariance_samples_are_estimates,
                k_scale,
                max_order,
                &row.sigma_trace,
                &row.value
            );
            if(std::isfinite(row.chi2))
                chi2_min = std::min(chi2_min, row.chi2);
            rows[
                static_cast<std::size_t>(ix * grid_points + iy)
            ] = row;
        }

        int done = 0;
        #pragma omp atomic capture
        done = ++completed_rows;
        if(done == 1 || done == grid_points || done % report_every == 0) {
            #pragma omp critical(print_direct_vn_chi2_progress)
            {
            std::cout << "[" << label << "] chi2 scan row "
                      << done << "/" << grid_points
                      << " finished\n";
            }
        }
    }

    std::ofstream out(filename);
    if(!out.is_open())
        throw std::runtime_error("Cannot open " + filename);

    out << std::setprecision(17);
    out << "# direct-vn generating-function covariance chi2 scan\n"
        << "# definition "
        << (max_order >= 0 ? "truncated-J0" : "full-J0") << "\n"
        << "# covariance_source "
        << (covariance_samples_are_estimates ? "bootstrap" : "subsample")
        << "\n"
        << "# mode " << (use_explicit_bounds ? "global" : "local") << "\n"
        << "# center_Re " << center.real()
        << " center_Im " << center.imag()
        << " half_width " << half_width
        << " re_min " << scan_re_min
        << " re_max " << scan_re_max
        << " im_min " << scan_im_min
        << " im_max " << scan_im_max
        << " grid_points " << grid_points
        << " chi2_min " << chi2_min << "\n"
        << "# contours: delta_chi2 1sigma=2.30 2sigma=6.18 3sigma=11.83\n"
        << "# Re(z) Im(z) chi2 delta_chi2 sigma_trace Re(G) Im(G)\n";

    for(const GridRow& row : rows) {
        out << row.re << " "
            << row.im << " "
            << row.chi2 << " "
            << (row.chi2 - chi2_min) << " "
            << row.sigma_trace << " "
            << row.value.real() << " "
            << row.value.imag() << "\n";
    }
}

class DirectVnLYZCachedEvaluator {
public:
    DirectVnLYZCachedEvaluator(
        const DirectVnSample& sample,
        double k_scale,
        int max_order)
        : sample_(&sample), k_scale_(k_scale), max_order_(max_order) {}

    const MCGValue& Eval(const double* x) const {
        if(!cache_valid_ || x[0] != last_re_ || x[1] != last_im_) {
            last_re_ = x[0];
            last_im_ = x[1];
            if(max_order_ >= 0) {
                cached_value_ = EvaluateDirectVnLYZTruncated(
                    cd(last_re_, last_im_),
                    *sample_,
                    k_scale_,
                    max_order_
                );
            } else {
                cached_value_ = EvaluateDirectVnJ0(
                    cd(last_re_, last_im_),
                    *sample_
                );
            }
            cache_valid_ = true;
        }
        return cached_value_;
    }

private:
    const DirectVnSample* sample_ = nullptr;
    double k_scale_ = 1.0;
    int max_order_ = 0;
    mutable bool cache_valid_ = false;
    mutable double last_re_ = 0.0;
    mutable double last_im_ = 0.0;
    mutable MCGValue cached_value_;
};

class ReDirectVnLYZGrad : public ROOT::Math::IGradientFunctionMultiDim {
public:
    explicit ReDirectVnLYZGrad(const DirectVnLYZCachedEvaluator& evaluator)
        : evaluator_(&evaluator) {}

    unsigned int NDim() const override { return 2; }

    ROOT::Math::IBaseFunctionMultiDim* Clone() const override {
        return new ReDirectVnLYZGrad(*this);
    }

private:
    double DoEval(const double* x) const override {
        return std::real(evaluator_->Eval(x).G);
    }

    double DoDerivative(const double* x, unsigned int icoord) const override {
        const cd dGdk = evaluator_->Eval(x).dGdk;
        return (icoord == 0) ? std::real(dGdk) : -std::imag(dGdk);
    }

    const DirectVnLYZCachedEvaluator* evaluator_;
};

class ImDirectVnLYZGrad : public ROOT::Math::IGradientFunctionMultiDim {
public:
    explicit ImDirectVnLYZGrad(const DirectVnLYZCachedEvaluator& evaluator)
        : evaluator_(&evaluator) {}

    unsigned int NDim() const override { return 2; }

    ROOT::Math::IBaseFunctionMultiDim* Clone() const override {
        return new ImDirectVnLYZGrad(*this);
    }

private:
    double DoEval(const double* x) const override {
        return std::imag(evaluator_->Eval(x).G);
    }

    double DoDerivative(const double* x, unsigned int icoord) const override {
        const cd dGdk = evaluator_->Eval(x).dGdk;
        return (icoord == 0) ? std::imag(dGdk) : std::real(dGdk);
    }

    const DirectVnLYZCachedEvaluator* evaluator_;
};

RootResult FindDirectVnLYZZero(
    const DirectVnSample& sample,
    double k_scale,
    int max_order,
    double x0_re,
    double x0_im)
{
    RootResult result;

    try {
        auto t1 = std::chrono::high_resolution_clock::now();
        ROOT::Math::GSLMultiRootFinder finder(
            ROOT::Math::GSLMultiRootFinder::kHybridSJ
        );

        DirectVnLYZCachedEvaluator evaluator(sample, k_scale, max_order);
        ReDirectVnLYZGrad f_re(evaluator);
        ImDirectVnLYZGrad f_im(evaluator);
        finder.AddFunction(f_re);
        finder.AddFunction(f_im);

        double x0[2] = {x0_re, x0_im};
        const bool ok = finder.Solve(x0, 1000, 1e-10, 1e-10);
        const double* root = finder.X();
        const double* fval = finder.FVal();

        result.ok = ok;
        result.status = finder.Status();
        if(root && fval) {
            result.k = {root[0], root[1]};
            result.F = {fval[0], fval[1]};
        }

        auto t2 = std::chrono::high_resolution_clock::now();
        result.time_ms =
            std::chrono::duration<double, std::milli>(t2 - t1).count();
    }
    catch(const std::exception&) {
        result.ok = false;
        result.status = -999;
    }

    return result;
}

std::vector<RootResult> FindDirectVnLYZZerosFromStarts(
    const DirectVnSample& sample,
    double k_scale,
    int max_order,
    const std::vector<std::pair<double, double>>& start_points)
{
    std::vector<RootResult> roots;

    double max_root_size = 0.0;
    for(const auto& p : start_points) {
        max_root_size = std::max(
            max_root_size,
            std::sqrt(p.first * p.first + p.second * p.second)
        );
    }
    max_root_size *= 2.0;
    if(max_root_size <= 0.0)
        max_root_size = std::numeric_limits<double>::infinity();

    for(const auto& p : start_points) {
        RootResult r = FindDirectVnLYZZero(
            sample,
            k_scale,
            max_order,
            p.first,
            p.second
        );
        if(IsGoodMCGRoot(r, max_root_size))
            roots.push_back(r);
    }

    const double duplicate_tol = 1e-6;
    std::vector<RootResult> unique_roots;
    for(const auto& r : roots) {
        bool duplicate = false;
        for(const auto& u : unique_roots) {
            if(std::abs(r.k - u.k) < duplicate_tol) {
                duplicate = true;
                break;
            }
        }
        if(!duplicate)
            unique_roots.push_back(r);
    }

    std::sort(
        unique_roots.begin(),
        unique_roots.end(),
        [](const RootResult& a, const RootResult& b) {
            return std::abs(a.k) < std::abs(b.k);
        }
    );

    return unique_roots;
}

// ------------------------------------------------------------
// Complex Struve H_nu series for nu = 0 or 1
// H_nu(z) = sum_m (-1)^m (z/2)^(2m+nu+1)
//           / [ Gamma(m+3/2) Gamma(m+nu+3/2) ]
// ------------------------------------------------------------
cd struveH_series(int nu, cd z, double tol = 1e-14, int max_iter = 10000)
{
    cd halfz = 0.5 * z;
    cd sum = 0.0;

    for(int m = 0; m < max_iter; ++m)
    {
        double sign = (m % 2 == 0) ? 1.0 : -1.0;

        double denom =
            std::tgamma(m + 1.5) *
            std::tgamma(m + nu + 1.5);

        cd term = sign * std::pow(halfz, 2*m + nu + 1) / denom;
        sum += term;

        if(std::abs(term) < tol * std::max(1.0, std::abs(sum)))
            break;
    }

    return sum;
}

// ------------------------------------------------------------
// C++ version of your triangleG_scalar(k,a,b)
// ------------------------------------------------------------
cd triangleG(cd k, double a, double b)
{
    cd ak = a * k;
    cd bk = b * k;

    cd J0a = besselJ0_series(ak);
    cd J1a = besselJ1_series(ak);
    cd J0b = besselJ0_series(bk);
    cd J1b = besselJ1_series(bk);

    cd H0a = struveH_series(0, ak);
    cd H1a = struveH_series(1, ak);
    cd H0b = struveH_series(0, bk);
    cd H1b = struveH_series(1, bk);

    cd result =
        (2.0 * J1a) / (b * k)
        +
        (
            (a * J1a * (-2.0 + b*k*M_PI*H0a)) / (b*k)
            + (J1b * (2.0 - b*k*M_PI*H0b)) / k
            + a * J0a * (2.0 - M_PI*H1a)
            + b * J0b * (-2.0 + M_PI*H1b)
        ) / (a - b);

    return result;
}

// ------------------------------------------------------------
// Q_n(theta) = sum_i cos[n(phi_i - theta)]
// ------------------------------------------------------------
double Qntheta(int n, const std::vector<double>& phiEvent, double theta)
{
    double Q = 0.0;

    for(double phi : phiEvent)
        Q += std::cos(n * (phi - theta));

    return Q;
}

cd QnVector(int n, const std::vector<double>& phiEvent)
{
    cd Q(0.0, 0.0);

    for(double phi : phiEvent)
        Q += std::polar(1.0, n * phi);

    return Q;
}

struct EventCorrelations {
    double two = NAN;
    double four = NAN;
    double six = NAN;
    double eight = NAN;
    double ten = NAN;
};

struct QCorrelationTerm {
    long long coefficient = 0;
    std::vector<int> harmonic_sums;
};

long long PartitionMobiusFactor(std::size_t block_size)
{
    long long factor = (block_size % 2 == 0) ? -1 : 1;
    for(std::size_t i = 2; i < block_size; ++i)
        factor *= static_cast<long long>(i);
    return factor;
}

std::vector<QCorrelationTerm> BuildQCorrelationTerms(int pairs)
{
    std::vector<int> harmonics(
        static_cast<std::size_t>(2 * pairs),
        -1
    );
    std::fill(harmonics.begin(), harmonics.begin() + pairs, 1);

    std::map<std::vector<int>, long long> combined_terms;
    std::vector<std::vector<int>> blocks;

    auto visit_partitions = [&](auto&& self, int index) -> void {
        if(index == static_cast<int>(harmonics.size())) {
            std::vector<int> harmonic_sums;
            long long coefficient = 1;
            for(const auto& block : blocks) {
                int sum = 0;
                for(int member : block)
                    sum += harmonics[static_cast<std::size_t>(member)];
                harmonic_sums.push_back(sum);
                coefficient *= PartitionMobiusFactor(block.size());
            }
            std::sort(harmonic_sums.begin(), harmonic_sums.end());
            combined_terms[harmonic_sums] += coefficient;
            return;
        }

        const std::size_t existing_blocks = blocks.size();
        for(std::size_t iblock = 0; iblock < existing_blocks; ++iblock) {
            blocks[iblock].push_back(index);
            self(self, index + 1);
            blocks[iblock].pop_back();
        }

        blocks.push_back({index});
        self(self, index + 1);
        blocks.pop_back();
    };
    visit_partitions(visit_partitions, 0);

    std::vector<QCorrelationTerm> terms;
    for(auto& entry : combined_terms) {
        if(entry.second != 0)
            terms.push_back({entry.second, std::move(entry.first)});
    }
    return terms;
}

double ComputeQVectorCorrelation(
    int pairs,
    const std::vector<cd>& q_vectors,
    int multiplicity
)
{
    const int particles = 2 * pairs;
    if(multiplicity < particles)
        return NAN;

    static const std::vector<std::vector<QCorrelationTerm>> terms = {
        {},
        BuildQCorrelationTerms(1),
        BuildQCorrelationTerms(2),
        BuildQCorrelationTerms(3),
        BuildQCorrelationTerms(4),
        BuildQCorrelationTerms(5)
    };

    cd numerator(0.0, 0.0);
    for(const QCorrelationTerm& term : terms[static_cast<std::size_t>(pairs)]) {
        cd product(static_cast<double>(term.coefficient), 0.0);
        for(int harmonic_sum : term.harmonic_sums)
            product *= q_vectors[static_cast<std::size_t>(harmonic_sum + 5)];
        numerator += product;
    }

    double denominator = 1.0;
    for(int i = 0; i < particles; ++i)
        denominator *= static_cast<double>(multiplicity - i);

    return std::real(numerator) / denominator;
}

EventCorrelations ComputeEventCorrelations(
    int n,
    const std::vector<double>& phiEvent
)
{
    EventCorrelations correlations;
    const int M = static_cast<int>(phiEvent.size());
    std::vector<cd> q_vectors(11, cd(0.0, 0.0));
    q_vectors[5] = cd(static_cast<double>(M), 0.0);

    for(double phi : phiEvent) {
        const cd unit = std::polar(1.0, n * phi);
        cd power(1.0, 0.0);
        for(int harmonic = 1; harmonic <= 5; ++harmonic) {
            power *= unit;
            q_vectors[static_cast<std::size_t>(harmonic + 5)] += power;
        }
    }
    for(int harmonic = 1; harmonic <= 5; ++harmonic) {
        q_vectors[static_cast<std::size_t>(5 - harmonic)] =
            std::conj(q_vectors[static_cast<std::size_t>(5 + harmonic)]);
    }

    correlations.two = ComputeQVectorCorrelation(1, q_vectors, M);
    correlations.four = ComputeQVectorCorrelation(2, q_vectors, M);
    correlations.six = ComputeQVectorCorrelation(3, q_vectors, M);
    correlations.eight = ComputeQVectorCorrelation(4, q_vectors, M);
    correlations.ten = ComputeQVectorCorrelation(5, q_vectors, M);
    return correlations;
}

void WriteEventCorrelations(
    const std::string& filename,
    int n,
    int event_multiplicity,
    const std::vector<EventCorrelations>& correlations
)
{
    std::vector<char> output_buffer(1 << 20);
    std::ofstream out;
    out.rdbuf()->pubsetbuf(output_buffer.data(), output_buffer.size());
    out.open(filename);

    if(!out.is_open())
        throw std::runtime_error("Cannot open " + filename);

    out << std::setprecision(17);
    out << "# ievent  n  M  <2>  <4>  <6>  <8>  <10>\n";

    for(std::size_t i = 0; i < correlations.size(); ++i) {
        const auto& c = correlations[i];

        out << i << " "
            << n << " "
            << event_multiplicity << " "
            << c.two << " "
            << c.four << " "
            << c.six << " "
            << c.eight << " "
            << c.ten << "\n";

        if((i + 1) % 100000 == 0 || i + 1 == correlations.size()) {
            std::cout << "  wrote " << (i + 1) << "/"
                      << correlations.size() << " rows to "
                      << filename << "\n" << std::flush;
        }
        if(!out.good()) {
            throw std::runtime_error("Failed while writing " + filename);
        }
    }
}

double EventCorrelationAtOrder(const EventCorrelations& correlations, int order)
{
    if(order == 2) return correlations.two;
    if(order == 4) return correlations.four;
    if(order == 6) return correlations.six;
    if(order == 8) return correlations.eight;
    if(order == 10) return correlations.ten;
    return NAN;
}

void WriteCorrelationCoefficientEstimates(
    const std::string& filename,
    int event_multiplicity,
    int Nsub,
    const std::vector<EventCorrelations>& correlations
)
{
    std::ofstream out(filename);
    if(!out.is_open())
        throw std::runtime_error("Cannot open " + filename);

    out << std::setprecision(17);
    out << "# order  mean_correlation  correlation_standard_error"
        << "  Re(G_coefficient)  G_coefficient_standard_error\n";
    out << "0 1 0 1 0\n";

    double multiplicity_power = 1.0;
    double four_power = 1.0;
    double factorial = 1.0;

    for(int order = 2; order <= 10; order += 2) {
        const int pairs = order / 2;
        multiplicity_power *=
            static_cast<double>(event_multiplicity) *
            static_cast<double>(event_multiplicity);
        four_power *= 4.0;
        factorial *= static_cast<double>(pairs);
        const double sign = (pairs % 2 == 0) ? 1.0 : -1.0;
        const double coefficient_factor =
            sign * multiplicity_power /
            (four_power * factorial * factorial);

        std::vector<double> subsample_sums(static_cast<std::size_t>(Nsub), 0.0);
        std::vector<double> subsample_counts(static_cast<std::size_t>(Nsub), 0.0);
        double sum = 0.0;
        double count = 0.0;

        for(std::size_t ievent = 0; ievent < correlations.size(); ++ievent) {
            const double value =
                EventCorrelationAtOrder(correlations[ievent], order);
            if(!std::isfinite(value))
                continue;

            sum += value;
            count += 1.0;
            const std::size_t isub = ievent % static_cast<std::size_t>(Nsub);
            subsample_sums[isub] += value;
            subsample_counts[isub] += 1.0;
        }

        const double mean = count > 0.0 ? sum / count : NAN;
        std::vector<double> subsample_means;
        for(int isub = 0; isub < Nsub; ++isub) {
            if(subsample_counts[static_cast<std::size_t>(isub)] > 0.0) {
                subsample_means.push_back(
                    subsample_sums[static_cast<std::size_t>(isub)] /
                    subsample_counts[static_cast<std::size_t>(isub)]
                );
            }
        }

        double standard_error = NAN;
        if(subsample_means.size() > 1) {
            double subsample_mean = 0.0;
            for(double value : subsample_means)
                subsample_mean += value;
            subsample_mean /= static_cast<double>(subsample_means.size());

            double variance_sum = 0.0;
            for(double value : subsample_means) {
                const double delta = value - subsample_mean;
                variance_sum += delta * delta;
            }
            standard_error = std::sqrt(
                variance_sum /
                (
                    static_cast<double>(subsample_means.size()) *
                    static_cast<double>(subsample_means.size() - 1)
                )
            );
        }

        out << order << " "
            << mean << " "
            << standard_error << " "
            << coefficient_factor * mean << " "
            << std::abs(coefficient_factor) * standard_error << "\n";
    }
}

int CorrelationIndexForOrder(int order)
{
    if(order == 2) return 0;
    if(order == 4) return 1;
    if(order == 6) return 2;
    if(order == 8) return 3;
    if(order == 10) return 4;
    return -1;
}

double EventCorrelationAtIndex(const EventCorrelations& correlations, int index)
{
    if(index == 0) return correlations.two;
    if(index == 1) return correlations.four;
    if(index == 2) return correlations.six;
    if(index == 3) return correlations.eight;
    if(index == 4) return correlations.ten;
    return NAN;
}

void AddEventCorrelationsToSubsampleSums(
    const EventCorrelations& correlations,
    std::vector<double>& sums,
    std::vector<double>& counts
)
{
    for(int index = 0; index < 5; ++index) {
        const double value = EventCorrelationAtIndex(correlations, index);
        if(std::isfinite(value)) {
            sums[static_cast<std::size_t>(index)] += value;
            counts[static_cast<std::size_t>(index)] += 1.0;
        }
    }
}

void WriteCorrelationCoefficientEstimatesFromSubsamples(
    const std::string& filename,
    int event_multiplicity,
    const std::vector<std::vector<double>>& subsample_sums,
    const std::vector<std::vector<double>>& subsample_counts
)
{
    std::ofstream out(filename);
    if(!out.is_open())
        throw std::runtime_error("Cannot open " + filename);

    out << std::setprecision(17);
    out << "# order  mean_correlation  correlation_standard_error"
        << "  Re(G_coefficient)  G_coefficient_standard_error\n";
    out << "0 1 0 1 0\n";

    double multiplicity_power = 1.0;
    double four_power = 1.0;
    double factorial = 1.0;

    for(int order = 2; order <= 10; order += 2) {
        const int index = CorrelationIndexForOrder(order);
        const int pairs = order / 2;
        multiplicity_power *=
            static_cast<double>(event_multiplicity) *
            static_cast<double>(event_multiplicity);
        four_power *= 4.0;
        factorial *= static_cast<double>(pairs);
        const double sign = (pairs % 2 == 0) ? 1.0 : -1.0;
        const double coefficient_factor =
            sign * multiplicity_power /
            (four_power * factorial * factorial);

        double sum = 0.0;
        double count = 0.0;
        std::vector<double> subsample_means;

        for(std::size_t isub = 0; isub < subsample_sums.size(); ++isub) {
            const double sub_count =
                subsample_counts[isub][static_cast<std::size_t>(index)];
            const double sub_sum =
                subsample_sums[isub][static_cast<std::size_t>(index)];
            sum += sub_sum;
            count += sub_count;
            if(sub_count > 0.0)
                subsample_means.push_back(sub_sum / sub_count);
        }

        const double mean = count > 0.0 ? sum / count : NAN;

        double standard_error = NAN;
        if(subsample_means.size() > 1) {
            double subsample_mean = 0.0;
            for(double value : subsample_means)
                subsample_mean += value;
            subsample_mean /= static_cast<double>(subsample_means.size());

            double variance_sum = 0.0;
            for(double value : subsample_means) {
                const double delta = value - subsample_mean;
                variance_sum += delta * delta;
            }
            standard_error = std::sqrt(
                variance_sum /
                (
                    static_cast<double>(subsample_means.size()) *
                    static_cast<double>(subsample_means.size() - 1)
                )
            );
        }

        out << order << " "
            << mean << " "
            << standard_error << " "
            << coefficient_factor * mean << " "
            << std::abs(coefficient_factor) * standard_error << "\n";
    }
}

void BuildCorrelationTruncatedSubsamples(
    const std::vector<EventCorrelations>& correlations,
    int event_multiplicity,
    int Nsub,
    std::vector<std::vector<cd>>& subsample_coeff_sums,
    std::vector<double>& subsample_counts)
{
    if(event_multiplicity > 10) {
        throw std::runtime_error(
            "--write-corr-lyz currently supports event multiplicity M <= 10"
        );
    }

    const int max_order = 2 * std::min(5, event_multiplicity / 2);
    subsample_coeff_sums.assign(
        static_cast<std::size_t>(Nsub),
        std::vector<cd>(static_cast<std::size_t>(max_order + 1), cd(0.0, 0.0))
    );
    subsample_counts.assign(static_cast<std::size_t>(Nsub), 0.0);

    std::vector<double> coefficient_factors(
        static_cast<std::size_t>(max_order + 1),
        0.0
    );
    double multiplicity_power = 1.0;
    double four_power = 1.0;
    double factorial = 1.0;
    for(int order = 2; order <= max_order; order += 2) {
        const int pairs = order / 2;
        multiplicity_power *=
            static_cast<double>(event_multiplicity) *
            static_cast<double>(event_multiplicity);
        four_power *= 4.0;
        factorial *= static_cast<double>(pairs);
        const double sign = (pairs % 2 == 0) ? 1.0 : -1.0;
        coefficient_factors[static_cast<std::size_t>(order)] =
            sign * multiplicity_power /
            (four_power * factorial * factorial);
    }

    for(std::size_t ievent = 0; ievent < correlations.size(); ++ievent) {
        const std::size_t isub = ievent % static_cast<std::size_t>(Nsub);
        auto& coeffs = subsample_coeff_sums[isub];

        coeffs[0] += cd(1.0, 0.0);
        bool valid_event = true;
        for(int order = 2; order <= max_order; order += 2) {
            const double corr =
                EventCorrelationAtOrder(correlations[ievent], order);
            if(!std::isfinite(corr)) {
                valid_event = false;
                break;
            }
            coeffs[static_cast<std::size_t>(order)] +=
                cd(coefficient_factors[static_cast<std::size_t>(order)] * corr, 0.0);
        }

        if(valid_event) {
            subsample_counts[isub] += 1.0;
        } else {
            coeffs[0] -= cd(1.0, 0.0);
            for(int order = 2; order <= max_order; order += 2) {
                const double corr =
                    EventCorrelationAtOrder(correlations[ievent], order);
                if(std::isfinite(corr)) {
                    coeffs[static_cast<std::size_t>(order)] -=
                        cd(coefficient_factors[static_cast<std::size_t>(order)] * corr, 0.0);
                }
            }
        }
    }
}

std::string AddFilenameSuffix(const std::string& filename, const std::string& suffix)
{
    if(filename.empty())
        return "";

    const std::size_t slash_pos = filename.find_last_of("/\\");
    const std::size_t dot_pos = filename.find_last_of('.');

    if(dot_pos == std::string::npos ||
       (slash_pos != std::string::npos && dot_pos < slash_pos)) {
        return filename + suffix;
    }

    return filename.substr(0, dot_pos) + suffix + filename.substr(dot_pos);
}

void FindAndWriteRootsForDataset(
    const std::string& label,
    const std::vector<double>& Q,
    int event_multiplicity,
    int q_bins,
    int Nsub,
    int Nresam,
    int ncore,
    unsigned int bootstrap_seed,
    const std::vector<std::pair<double, double>>& root_start_points,
    const std::string& nominal_output,
    const std::string& bootstrap_output,
    const std::string& root_diagnostics_output = "",
    const std::string& coefficient_error_output = "",
    int coefficient_error_grid = 0,
    double coefficient_error_half_width = -1.0,
    double coefficient_error_re_min = NAN,
    double coefficient_error_re_max = NAN,
    double coefficient_error_im_min = NAN,
    double coefficient_error_im_max = NAN
)
{
    const QBinLayout q_bin_layout = BuildQBinLayout(q_bins, event_multiplicity);
    const MCGSample nominal_sample = BuildQSample(Q, q_bin_layout);

    std::cout << "[" << label << "] Root search Q sample entries: "
              << nominal_sample.q.size()
              << " weighted points from "
              << nominal_sample.total_weight
              << " events\n";

    std::cout << "[" << label << "] Finding MC generating-function zeros from "
              << root_start_points.size() << " starts\n";

    std::vector<RootResult> mc_roots = FindMCGeneratingZerosFromStarts(
        nominal_sample,
        root_start_points,
        true
    );

    WriteMCGeneratingRoots(nominal_output, mc_roots);

    std::cout << "[" << label << "] Found " << mc_roots.size()
              << " nominal unique roots\n";
    for(std::size_t iroot = 0; iroot < mc_roots.size(); ++iroot) {
        const auto& r = mc_roots[iroot];
        std::cout << "[" << label << "] root " << iroot
                  << ": k = " << r.k
                  << ", G(k) = " << r.F
                  << ", status = " << r.status
                  << "\n";
    }

    std::vector<std::vector<double>> Q_subsample_bins;
    std::vector<std::vector<double>> Q_exact_subsamples;

    if(q_bins > 0) {
        Q_subsample_bins = BuildQSubsampleBins(Q, Nsub, q_bin_layout);
    } else {
        Q_exact_subsamples = BuildExactQSubsamples(Q, Nsub);
    }

    std::vector<std::vector<RootResult>> bootstrap_roots;
    bootstrap_roots.resize(static_cast<std::size_t>(Nresam));
    std::vector<std::vector<RootDiagnostic>> diagnostics_by_resample(
        static_cast<std::size_t>(Nresam)
    );

    std::cout << "[" << label << "] Starting bootstrap root search with "
              << Nresam << " resamples and "
              << Nsub << " Q subsamples using "
              << ncore << " threads\n";

    omp_set_num_threads(ncore);
    int completed_bootstrap = 0;

    #pragma omp parallel for schedule(dynamic)
    for(int ires = 0; ires < Nresam; ++ires) {
        TRandom3 bootstrap_rng(bootstrap_seed + static_cast<unsigned int>(ires));
        MCGSample Q_bootstrap =
            (q_bins > 0)
                ? BootstrapQSampleFromSubsampleBins(
                      Q_subsample_bins,
                      q_bin_layout,
                      bootstrap_rng
                  )
                : BootstrapExactQSampleFromSubsamples(
                      Q_exact_subsamples,
                      bootstrap_rng
                  );

        std::vector<RootResult> roots = FindMCGeneratingZerosFromStarts(
            Q_bootstrap,
            root_start_points,
            true
        );

        bootstrap_roots[static_cast<std::size_t>(ires)] = roots;

        int done = 0;
        #pragma omp atomic capture
        done = ++completed_bootstrap;

        #pragma omp critical(print_bootstrap_progress)
        {
            std::cout << "[" << label << "] bootstrap "
                      << done << "/" << Nresam
                      << " finished: ires=" << ires
                      << ", roots=" << roots.size()
                      << "\n";
        }
    }

    WriteBootstrapMCGeneratingRoots(
        bootstrap_output,
        bootstrap_roots
    );
}

void FindAndWriteRootsForBinnedDataset(
    const std::string& label,
    const std::vector<std::vector<double>>& Q_subsample_bins,
    const QBinLayout& q_bin_layout,
    int Nresam,
    int ncore,
    unsigned int bootstrap_seed,
    const std::vector<std::pair<double, double>>& root_start_points,
    const std::string& nominal_output,
    const std::string& bootstrap_output
)
{
    const MCGSample nominal_sample =
        BuildQSampleFromSubsampleBins(Q_subsample_bins, q_bin_layout);

    std::cout << "[" << label << "] Root search Q sample entries: "
              << nominal_sample.q.size()
              << " weighted points from "
              << nominal_sample.total_weight
              << " events\n";

    std::cout << "[" << label << "] Finding MC generating-function zeros from "
              << root_start_points.size() << " starts\n";

    std::vector<RootResult> mc_roots = FindMCGeneratingZerosFromStarts(
        nominal_sample,
        root_start_points,
        true
    );

    WriteMCGeneratingRoots(nominal_output, mc_roots);

    std::cout << "[" << label << "] Found " << mc_roots.size()
              << " nominal unique roots\n";
    for(std::size_t iroot = 0; iroot < mc_roots.size(); ++iroot) {
        const auto& r = mc_roots[iroot];
        std::cout << "[" << label << "] root " << iroot
                  << ": k = " << r.k
                  << ", G(k) = " << r.F
                  << ", status = " << r.status
                  << "\n";
    }

    std::vector<std::vector<RootResult>> bootstrap_roots;
    bootstrap_roots.resize(static_cast<std::size_t>(Nresam));

    std::cout << "[" << label << "] Starting bootstrap root search with "
              << Nresam << " resamples and "
              << Q_subsample_bins.size()
              << " Q subsamples using "
              << ncore << " threads\n";

    omp_set_num_threads(ncore);
    int completed_bootstrap = 0;

    #pragma omp parallel for schedule(dynamic)
    for(int ires = 0; ires < Nresam; ++ires) {
        TRandom3 bootstrap_rng(bootstrap_seed + static_cast<unsigned int>(ires));
        MCGSample Q_bootstrap =
            BootstrapQSampleFromSubsampleBins(
                Q_subsample_bins,
                q_bin_layout,
                bootstrap_rng
            );

        std::vector<RootResult> roots = FindMCGeneratingZerosFromStarts(
            Q_bootstrap,
            root_start_points,
            true
        );

        bootstrap_roots[static_cast<std::size_t>(ires)] = roots;

        int done = 0;
        #pragma omp atomic capture
        done = ++completed_bootstrap;

        #pragma omp critical(print_bootstrap_progress)
        {
            std::cout << "[" << label << "] bootstrap "
                      << done << "/" << Nresam
                      << " finished: ires=" << ires
                      << ", roots=" << roots.size()
                      << "\n";
        }
    }

    WriteBootstrapMCGeneratingRoots(
        bootstrap_output,
        bootstrap_roots
    );
}

void FindAndWriteProductRootsForDataset(
    const std::string& label,
    const std::vector<std::vector<cd>>& subsample_coeff_sums,
    const std::vector<double>& subsample_counts,
    const std::vector<double>& coefficient_factors,
    bool remove_odd_coefficients,
    int Nresam,
    int ncore,
    unsigned int bootstrap_seed,
    const std::vector<std::pair<double, double>>& root_start_points,
    const std::string& nominal_output,
    const std::string& bootstrap_output,
    const std::string& root_diagnostics_output = "",
    const std::string& coefficient_error_output = "",
    int coefficient_error_grid = 0,
    double coefficient_error_half_width = -1.0,
    double coefficient_error_re_min = NAN,
    double coefficient_error_re_max = NAN,
    double coefficient_error_im_min = NAN,
    double coefficient_error_im_max = NAN
)
{
    ProductGSample nominal_before =
        BuildProductSampleFromSubsamples(subsample_coeff_sums, subsample_counts);
    ProductGSample nominal_sample = nominal_before;
    TransformProductSample(
        nominal_sample,
        coefficient_factors,
        remove_odd_coefficients
    );
    const bool save_exact_coefficients = label.rfind("exact-", 0) == 0;
    const bool save_product_coefficients =
        label.rfind("product-", 0) == 0 &&
        label.rfind("product-falling-", 0) != 0 &&
        label.rfind("product-correction-", 0) != 0 &&
        label.rfind("product-correction2-", 0) != 0;
    const bool save_falling_coefficients =
        label.rfind("product-falling-", 0) == 0;
    const bool save_coefficient_error_scan =
        !coefficient_error_output.empty();
    if(save_exact_coefficients) {
        WriteProductCoefficientComparison(
            AddFilenameSuffix(nominal_output, "_coefficients"),
            nominal_before,
            nominal_sample,
            coefficient_factors
        );
    }
    if(save_product_coefficients || save_falling_coefficients) {
        WriteProductCoefficients(
            AddFilenameSuffix(nominal_output, "_coefficients"),
            nominal_sample
        );
    }

    std::cout << "[" << label << "] Polynomial generating-function degree: "
              << (nominal_sample.coeffs.empty()
                      ? 0
                      : nominal_sample.coeffs.size() - 1)
              << ", events = " << nominal_sample.total_weight
              << ", odd coefficients = "
              << (remove_odd_coefficients ? "removed" : "kept")
              << "\n";
    if(label.rfind("product-falling-subevents-", 0) == 0) {
        std::cout << "[" << label
                  << "] eta-subevent grouped coefficients: order p<=Neta partitions eta windows into p contiguous groups; higher orders use ordinary falling coefficients\n";
    }

    if(label.rfind("exact-", 0) == 0 && !nominal_sample.coeffs.empty()) {
        const double multiplicity =
            static_cast<double>(nominal_sample.coeffs.size() - 1);
        double multiplicity_power = 1.0;
        double four_power = 1.0;
        double factorial = 1.0;

        for(std::size_t order = 2; order < nominal_sample.coeffs.size(); order += 2) {
            const std::size_t pairs = order / 2;
            multiplicity_power *= multiplicity * multiplicity;
            four_power *= 4.0;
            factorial *= static_cast<double>(pairs);
            const double sign = (pairs % 2 == 0) ? 1.0 : -1.0;
            const cd inferred_moment =
                nominal_sample.coeffs[order] *
                sign *
                four_power *
                factorial *
                factorial /
                multiplicity_power;

            std::vector<double> subsample_moments;
            for(std::size_t isub = 0; isub < subsample_coeff_sums.size(); ++isub) {
                if(subsample_counts[isub] <= 0.0)
                    continue;
                const cd coefficient =
                    subsample_coeff_sums[isub][order] /
                    subsample_counts[isub] *
                    coefficient_factors[order];
                subsample_moments.push_back(std::real(
                    coefficient *
                    sign *
                    four_power *
                    factorial *
                    factorial /
                    multiplicity_power
                ));
            }

            double standard_error = NAN;
            if(subsample_moments.size() > 1) {
                double mean = 0.0;
                for(double value : subsample_moments)
                    mean += value;
                mean /= static_cast<double>(subsample_moments.size());

                double variance_sum = 0.0;
                for(double value : subsample_moments) {
                    const double delta = value - mean;
                    variance_sum += delta * delta;
                }
                standard_error = std::sqrt(
                    variance_sum /
                    (
                        static_cast<double>(subsample_moments.size()) *
                        static_cast<double>(subsample_moments.size() - 1)
                    )
                );
            }

            std::cout << "[" << label << "] inferred <v^" << order << "> = "
                      << inferred_moment
                      << " +/- " << standard_error
                      << "\n";
        }
    }

    std::cout << "[" << label << "] Finding product generating-function zeros from "
              << root_start_points.size() << " starts\n";

    std::vector<RootResult> roots =
        FindProductGeneratingZerosFromStarts(nominal_sample, root_start_points);

    WriteMCGeneratingRoots(nominal_output, roots);

    std::cout << "[" << label << "] Found " << roots.size()
              << " nominal unique roots\n";
    for(std::size_t iroot = 0; iroot < roots.size(); ++iroot) {
        const auto& r = roots[iroot];
        std::cout << "[" << label << "] root " << iroot
                  << ": z = " << r.k
                  << ", G(z) = " << r.F
                  << ", status = " << r.status
                  << "\n";
    }

    std::vector<std::vector<RootResult>> bootstrap_roots;
    bootstrap_roots.resize(static_cast<std::size_t>(Nresam));
    std::vector<std::vector<RootDiagnostic>> diagnostics_by_resample(
        static_cast<std::size_t>(Nresam)
    );
    std::vector<ProductGSample> bootstrap_before;
    std::vector<ProductGSample> bootstrap_after;
    if(save_exact_coefficients) {
        bootstrap_before.resize(static_cast<std::size_t>(Nresam));
    }
    if(save_exact_coefficients ||
       save_product_coefficients ||
       save_falling_coefficients ||
       save_coefficient_error_scan) {
        bootstrap_after.resize(static_cast<std::size_t>(Nresam));
    }

    std::cout << "[" << label << "] Starting product bootstrap root search with "
              << Nresam << " resamples using "
              << ncore << " threads\n";

    omp_set_num_threads(ncore);
    int completed_bootstrap = 0;

    #pragma omp parallel for schedule(dynamic)
    for(int ires = 0; ires < Nresam; ++ires) {
        TRandom3 bootstrap_rng(bootstrap_seed + static_cast<unsigned int>(ires));
        ProductGSample bootstrap_before_sample =
            BootstrapProductSampleFromSubsamples(
                subsample_coeff_sums,
                subsample_counts,
                bootstrap_rng
            );
        ProductGSample bootstrap_sample = bootstrap_before_sample;
        TransformProductSample(
            bootstrap_sample,
            coefficient_factors,
            remove_odd_coefficients
        );
        if(save_exact_coefficients) {
            bootstrap_before[static_cast<std::size_t>(ires)] =
                bootstrap_before_sample;
        }
        if(save_exact_coefficients ||
           save_product_coefficients ||
           save_falling_coefficients ||
           save_coefficient_error_scan) {
            bootstrap_after[static_cast<std::size_t>(ires)] = bootstrap_sample;
        }
        std::vector<RootResult> local_roots =
            FindProductGeneratingZerosFromStarts(
                bootstrap_sample,
                root_start_points,
                ires,
                root_diagnostics_output.empty()
                    ? nullptr
                    : &diagnostics_by_resample[
                        static_cast<std::size_t>(ires)
                      ]
            );

        bootstrap_roots[static_cast<std::size_t>(ires)] = local_roots;

        int done = 0;
        #pragma omp atomic capture
        done = ++completed_bootstrap;

        #pragma omp critical(print_bootstrap_progress)
        {
            std::cout << "[" << label << "] bootstrap "
                      << done << "/" << Nresam
                      << " finished: ires=" << ires
                      << ", roots=" << local_roots.size()
                      << "\n";
        }
    }

    WriteBootstrapMCGeneratingRoots(
        bootstrap_output,
        bootstrap_roots
    );
    if(!root_diagnostics_output.empty()) {
        std::vector<RootDiagnostic> diagnostics;
        for(const auto& resample_diagnostics : diagnostics_by_resample) {
            diagnostics.insert(
                diagnostics.end(),
                resample_diagnostics.begin(),
                resample_diagnostics.end()
            );
        }
        WriteRootDiagnostics(root_diagnostics_output, diagnostics);
    }
    if(save_exact_coefficients) {
        WriteBootstrapProductCoefficientComparison(
            AddFilenameSuffix(bootstrap_output, "_coefficients"),
            bootstrap_before,
            bootstrap_after,
            coefficient_factors
        );
    }
    if(save_product_coefficients || save_falling_coefficients) {
        WriteBootstrapProductCoefficients(
            AddFilenameSuffix(bootstrap_output, "_coefficients"),
            bootstrap_after
        );
    }
    if(save_coefficient_error_scan) {
        WriteCoefficientChi2Scan(
            coefficient_error_output,
            nominal_sample,
            bootstrap_after,
            roots,
            root_start_points,
            remove_odd_coefficients,
            coefficient_error_grid,
            coefficient_error_half_width,
            coefficient_error_re_min,
            coefficient_error_re_max,
            coefficient_error_im_min,
            coefficient_error_im_max
        );
    }
}

void FindAndWriteDirectVnLYZRoots(
    const std::string& label,
    const std::vector<std::vector<double>>& vn_subsample_counts,
    double vn_max,
    double k_scale,
    int max_order,
    int Nresam,
    int ncore,
    unsigned int bootstrap_seed,
    const std::vector<std::pair<double, double>>& root_start_points,
    const std::string& nominal_output,
    const std::string& bootstrap_output,
    const std::string& function_error_output = "",
    const std::string& function_error_source = "subsample",
    int function_error_grid = 0,
    double function_error_half_width = -1.0,
    double function_error_re_min = NAN,
    double function_error_re_max = NAN,
    double function_error_im_min = NAN,
    double function_error_im_max = NAN)
{
    DirectVnSample nominal_sample =
        BuildDirectVnSampleFromSubsamples(vn_subsample_counts, vn_max);
    const bool save_function_error_scan =
        !function_error_output.empty() && function_error_source != "none";

    if(max_order >= 0) {
        std::cout << "[" << label << "] Finding sampled-vn LYZ zeros of "
                  << "truncated <J0(" << k_scale
                  << " z vn)> through order "
                  << max_order << " from "
                  << root_start_points.size() << " starts; events = "
                  << nominal_sample.total_weight << "\n";
    } else {
        std::cout << "[" << label << "] Finding sampled-vn zeros of "
                  << "full unscaled <J0(k vn)> from "
                  << root_start_points.size() << " starts; events = "
                  << nominal_sample.total_weight << "\n";
    }

    std::vector<RootResult> roots =
        FindDirectVnLYZZerosFromStarts(
            nominal_sample,
            k_scale,
            max_order,
            root_start_points
        );
    WriteMCGeneratingRoots(nominal_output, roots);

    std::cout << "[" << label << "] Found " << roots.size()
              << " nominal unique roots\n";
    const char* root_variable = (max_order >= 0) ? "z" : "k";
    for(std::size_t iroot = 0; iroot < roots.size(); ++iroot) {
        const auto& r = roots[iroot];
        std::cout << "[" << label << "] root " << iroot
                  << ": " << root_variable << " = " << r.k
                  << ", G(k) = " << r.F
                  << ", status = " << r.status
                  << "\n";
    }

    std::vector<std::vector<RootResult>> bootstrap_roots(
        static_cast<std::size_t>(Nresam)
    );
    std::vector<DirectVnSample> covariance_samples;
    if(save_function_error_scan && function_error_source == "subsample")
        covariance_samples =
            BuildDirectVnSamplesForSubsamples(vn_subsample_counts, vn_max);
    if(save_function_error_scan && function_error_source == "bootstrap")
        covariance_samples.resize(static_cast<std::size_t>(Nresam));

    std::cout << "[" << label << "] Starting bootstrap root search with "
              << Nresam << " resamples using "
              << ncore << " threads\n";

    omp_set_num_threads(ncore);
    int completed_bootstrap = 0;

    #pragma omp parallel for schedule(dynamic)
    for(int ires = 0; ires < Nresam; ++ires) {
        TRandom3 bootstrap_rng(bootstrap_seed + static_cast<unsigned int>(ires));
        DirectVnSample bootstrap_sample =
            BootstrapDirectVnSampleFromSubsamples(
                vn_subsample_counts,
                vn_max,
                bootstrap_rng
            );
        if(save_function_error_scan && function_error_source == "bootstrap")
            covariance_samples[static_cast<std::size_t>(ires)] =
                bootstrap_sample;

        std::vector<RootResult> local_roots =
            FindDirectVnLYZZerosFromStarts(
                bootstrap_sample,
                k_scale,
                max_order,
                root_start_points
            );

        bootstrap_roots[static_cast<std::size_t>(ires)] = local_roots;

        int done = 0;
        #pragma omp atomic capture
        done = ++completed_bootstrap;

        #pragma omp critical(print_vn_direct_bootstrap_progress)
        {
            std::cout << "[" << label << "] bootstrap "
                      << done << "/" << Nresam
                      << " finished: ires=" << ires
                      << ", roots=" << local_roots.size()
                      << "\n";
        }
    }

    WriteBootstrapMCGeneratingRoots(bootstrap_output, bootstrap_roots);

    if(save_function_error_scan) {
        std::cout << "[" << label << "] Writing direct-vn chi2 scan to "
                  << function_error_output << " using "
                  << covariance_samples.size()
                  << " " << function_error_source
                  << " samples and grid "
                  << function_error_grid << "x"
                  << function_error_grid << "\n";
        WriteDirectVnFunctionChi2Scan(
            function_error_output,
            label,
            nominal_sample,
            covariance_samples,
            function_error_source == "bootstrap",
            roots,
            root_start_points,
            k_scale,
            max_order,
            function_error_grid,
            function_error_half_width,
            function_error_re_min,
            function_error_re_max,
            function_error_im_min,
            function_error_im_max
        );
    }
}

// ------------------------------------------------------------
// Main implementation. The ROOT macro wrapper and executable main both call this.
// ------------------------------------------------------------
int RunTriangleG(const TriangleGConfig& cfg)
{
    std::cout << "Complex bessel: " << besselJ0_series(cd(1.35, 0.5)) << "\n";
    gStyle->SetOptStat(0);
    gsl_set_error_handler_off();
    gErrorIgnoreLevel = kWarning;

    TRandom3 rng(cfg.seed);

    const int M = (cfg.M_flow >= 0) ? cfg.M_flow : cfg.M;
    const int Nevents = cfg.Nevents;
    const int Nsub = cfg.Nsub;
    const int Nresam = cfg.Nresam;
    const int ncore = cfg.ncore;
    const int q_bins = cfg.q_bins;
    const int vn_bins = cfg.vn_bins;
    const int nonflow_q = cfg.nonflow_q;
    const int requested_M_nonflow =
        (nonflow_q > 0)
            ? (
                cfg.M_nonflow >= 0
                    ? cfg.M_nonflow
                    : (cfg.M_flow_set ? 0 : M)
              )
            : 0;
    const int nonflow_groups =
        (nonflow_q > 0)
            ? (
                cfg.nonflow_groups >= 0
                    ? cfg.nonflow_groups
                    : requested_M_nonflow / nonflow_q
              )
            : 0;
    const int nonflow_particles = nonflow_groups * nonflow_q;
    const int event_multiplicity = M + nonflow_particles;
    const bool has_flow = M > 0;
    const bool has_nonflow = nonflow_particles > 0;
    const bool has_combined = event_multiplicity > 0;
    if(!has_combined)
        throw std::runtime_error("Total combined multiplicity is zero; use --M-flow > 0 or enable nonflow particles");

    const double a = cfg.a;
    const double b = cfg.b;
    const double eta_max = cfg.eta_max;

    const int n = cfg.n;
    const double theta = cfg.theta;
    const bool use_exponential_roots =
        cfg.generating_function == "exponential" ||
        cfg.generating_function == "both" ||
        cfg.generating_function == "all";
    const bool use_product_roots =
        cfg.generating_function == "product" ||
        cfg.generating_function == "both" ||
        cfg.generating_function == "all";
    const bool use_product_falling_roots =
        cfg.generating_function == "product-falling" ||
        cfg.generating_function == "all";
    const bool use_product_falling_subevent_roots =
        cfg.generating_function == "product-falling-subevents" ||
        cfg.generating_function == "all";
    const bool use_product_correction_roots =
        cfg.generating_function == "product-correction" ||
        cfg.generating_function == "all";
    const bool use_product_correction2_roots =
        cfg.generating_function == "product-correction2" ||
        cfg.generating_function == "all";
    const bool use_modified_roots =
        cfg.generating_function == "modified" || cfg.generating_function == "all";
    const bool use_exact_roots =
        cfg.generating_function == "exact" || cfg.generating_function == "all";
    const bool need_raw_product_coefficients =
        use_product_roots ||
        use_product_falling_roots ||
        use_product_correction_roots ||
        use_product_correction2_roots ||
        use_exact_roots;
    const bool need_falling_subevent_coefficients =
        use_product_falling_subevent_roots;
    const bool need_stored_correlations =
        cfg.write_correlations ||
        cfg.write_correlation_truncated_lyz;
    const bool stream_exponential_q = use_exponential_roots && q_bins > 0;
    const bool store_exponential_q = use_exponential_roots && q_bins == 0;
    const QBinLayout flow_q_bin_layout =
        (stream_exponential_q && has_flow)
            ? BuildQBinLayout(q_bins, M)
            : QBinLayout();
    const QBinLayout nonflow_q_bin_layout =
        (stream_exponential_q && has_nonflow)
            ? BuildQBinLayout(q_bins, nonflow_particles)
            : QBinLayout();
    const QBinLayout combined_q_bin_layout =
        (stream_exponential_q && has_combined)
            ? BuildQBinLayout(q_bins, event_multiplicity)
            : QBinLayout();
    const bool need_direct_vn_roots =
        has_flow && (cfg.write_vn_lyz || cfg.write_vn_j0);
    const std::vector<double> flow_modified_factors =
        (use_modified_roots && has_flow)
            ? BuildModifiedProductFactors(M, cfg.modified_gl_nodes)
            : std::vector<double>();
    const std::vector<double> combined_modified_factors =
        use_modified_roots
            ? BuildModifiedProductFactors(event_multiplicity, cfg.modified_gl_nodes)
            : std::vector<double>();
    const std::vector<double> nonflow_modified_factors =
        (use_modified_roots && has_nonflow)
            ? BuildModifiedProductFactors(nonflow_particles, cfg.modified_gl_nodes)
            : std::vector<double>();
    const std::vector<double> flow_exact_factors =
        (use_exact_roots && has_flow)
            ? BuildExactProductFactors(M)
            : std::vector<double>();
    const std::vector<double> combined_exact_factors =
        use_exact_roots
            ? BuildExactProductFactors(event_multiplicity)
            : std::vector<double>();
    const std::vector<double> nonflow_exact_factors =
        (use_exact_roots && has_nonflow)
            ? BuildExactProductFactors(nonflow_particles)
            : std::vector<double>();
    const std::vector<double> flow_product_falling_factors =
        (use_product_falling_roots && has_flow)
            ? BuildExactProductFactors(M)
            : std::vector<double>();
    const std::vector<double> combined_product_falling_factors =
        use_product_falling_roots
            ? BuildExactProductFactors(event_multiplicity)
            : std::vector<double>();
    const std::vector<double> nonflow_product_falling_factors =
        (use_product_falling_roots && has_nonflow)
            ? BuildExactProductFactors(nonflow_particles)
            : std::vector<double>();
    const std::vector<double> flow_product_correction_factors =
        (use_product_correction_roots && has_flow)
            ? BuildProductDerivativeCorrectionFactors(M)
            : std::vector<double>();
    const std::vector<double> combined_product_correction_factors =
        use_product_correction_roots
            ? BuildProductDerivativeCorrectionFactors(event_multiplicity)
            : std::vector<double>();
    const std::vector<double> nonflow_product_correction_factors =
        (use_product_correction_roots && has_nonflow)
            ? BuildProductDerivativeCorrectionFactors(nonflow_particles)
            : std::vector<double>();
    const std::vector<double> flow_product_correction2_factors =
        (use_product_correction2_roots && has_flow)
            ? BuildProductDerivativeCorrection2Factors(M)
            : std::vector<double>();
    const std::vector<double> combined_product_correction2_factors =
        use_product_correction2_roots
            ? BuildProductDerivativeCorrection2Factors(event_multiplicity)
            : std::vector<double>();
    const std::vector<double> nonflow_product_correction2_factors =
        (use_product_correction2_roots && has_nonflow)
            ? BuildProductDerivativeCorrection2Factors(nonflow_particles)
            : std::vector<double>();
    const std::vector<double> no_coefficient_factors;

    std::cout << std::setprecision(17);
    std::cout << "Configuration:\n"
              << "  a = " << a << "\n"
              << "  b = " << b << "\n"
              << "  M_flow = " << M << "\n"
              << "  Nevents = " << Nevents << "\n"
              << "  Nsub = " << Nsub << "\n"
              << "  Nresam = " << Nresam << "\n"
              << "  ncore = " << ncore << "\n"
              << "  q_bins = " << q_bins << "\n"
              << "  vn_bins = " << vn_bins << "\n"
              << "  Neta = " << cfg.Neta << "\n"
              << "  generating function = " << cfg.generating_function << "\n"
              << "  modified N_GL = " << cfg.modified_gl_nodes << "\n"
              << "  product theta bins = " << cfg.product_theta_bins << "\n"
              << "  nonflow_q = " << nonflow_q << "\n"
              << "  nonflow_delta_phi = " << cfg.nonflow_delta << "\n"
              << "  nonflow_delta_eta = " << cfg.nonflow_delta_eta << "\n"
              << "  eta_max = " << eta_max << "\n"
              << "  nonflow_groups = " << nonflow_groups << "\n"
              << "  M_nonflow = " << nonflow_particles << "\n"
              << "  flow-only event multiplicity = " << M << "\n"
              << "  nonflow-only event multiplicity = " << nonflow_particles << "\n"
              << "  combined event multiplicity = " << event_multiplicity << "\n"
              << "  correlations output = " << cfg.correlations_output << "\n"
              << "  write event correlations = "
              << (cfg.write_correlations ? "yes" : "no") << "\n"
              << "  write direct sampled-vn LYZ = "
              << (cfg.write_vn_lyz ? "yes" : "no") << "\n"
              << "  write unscaled sampled-vn J0 = "
              << (cfg.write_vn_j0 ? "yes" : "no") << "\n"
              << "  write <2m>-truncated LYZ = "
              << (cfg.write_correlation_truncated_lyz ? "yes" : "no") << "\n"
              << "  root start points = "
              << cfg.root_start_points.size() << "\n";
    for(std::size_t istart = 0; istart < cfg.root_start_points.size(); ++istart) {
        std::cout << "    start " << istart << " = "
                  << cfg.root_start_points[istart].first
                  << " + i " << cfg.root_start_points[istart].second
                  << "\n";
    }

    cd test_k(0.3, 0.02588);
    std::cout << "triangleG(0.3 + 0.02588i, 0.04, 0.15) = "
              << triangleG(test_k, a, b) << "\n";

    std::vector<double> Q_flow;
    std::vector<double> Q_nonflow;
    std::vector<double> Q_combined;
    if(store_exponential_q) {
        Q_flow.resize(static_cast<std::size_t>(Nevents));
        Q_nonflow.resize(static_cast<std::size_t>(Nevents));
        Q_combined.resize(static_cast<std::size_t>(Nevents));
    }
    std::vector<EventCorrelations> flow_correlations;
    std::vector<EventCorrelations> nonflow_correlations;
    std::vector<EventCorrelations> combined_correlations;
    if(need_stored_correlations) {
        flow_correlations.resize(static_cast<std::size_t>(Nevents));
        nonflow_correlations.resize(static_cast<std::size_t>(Nevents));
        combined_correlations.resize(static_cast<std::size_t>(Nevents));
    }
    int completed_events = 0;

    std::vector<std::vector<std::vector<cd>>> flow_product_thread_sums;
    std::vector<std::vector<std::vector<cd>>> nonflow_product_thread_sums;
    std::vector<std::vector<std::vector<cd>>> combined_product_thread_sums;
    std::vector<std::vector<std::vector<cd>>> flow_product_thread_compensations;
    std::vector<std::vector<std::vector<cd>>> nonflow_product_thread_compensations;
    std::vector<std::vector<std::vector<cd>>> combined_product_thread_compensations;
    std::vector<std::vector<double>> product_thread_counts;
    std::vector<std::vector<std::vector<cd>>> flow_falling_subevent_thread_sums;
    std::vector<std::vector<std::vector<cd>>> nonflow_falling_subevent_thread_sums;
    std::vector<std::vector<std::vector<cd>>> combined_falling_subevent_thread_sums;
    std::vector<std::vector<std::vector<cd>>> flow_falling_subevent_thread_compensations;
    std::vector<std::vector<std::vector<cd>>> nonflow_falling_subevent_thread_compensations;
    std::vector<std::vector<std::vector<cd>>> combined_falling_subevent_thread_compensations;
    std::vector<std::vector<double>> falling_subevent_thread_counts;
    std::vector<std::vector<std::vector<cd>>> flow_modified_thread_sums;
    std::vector<std::vector<std::vector<cd>>> nonflow_modified_thread_sums;
    std::vector<std::vector<std::vector<cd>>> combined_modified_thread_sums;
    std::vector<std::vector<std::vector<cd>>> flow_modified_thread_compensations;
    std::vector<std::vector<std::vector<cd>>> nonflow_modified_thread_compensations;
    std::vector<std::vector<std::vector<cd>>> combined_modified_thread_compensations;
    std::vector<std::vector<double>> modified_thread_counts;
    std::vector<std::vector<std::vector<double>>> vn_thread_counts;
    std::vector<std::vector<std::vector<double>>> flow_correlation_thread_sums;
    std::vector<std::vector<std::vector<double>>> nonflow_correlation_thread_sums;
    std::vector<std::vector<std::vector<double>>> combined_correlation_thread_sums;
    std::vector<std::vector<std::vector<double>>> flow_correlation_thread_counts;
    std::vector<std::vector<std::vector<double>>> nonflow_correlation_thread_counts;
    std::vector<std::vector<std::vector<double>>> combined_correlation_thread_counts;
    std::vector<std::vector<std::vector<double>>> flow_q_thread_bins;
    std::vector<std::vector<std::vector<double>>> nonflow_q_thread_bins;
    std::vector<std::vector<std::vector<double>>> combined_q_thread_bins;

    flow_correlation_thread_sums.assign(
        static_cast<std::size_t>(ncore),
        std::vector<std::vector<double>>(
            static_cast<std::size_t>(Nsub),
            std::vector<double>(5, 0.0)
        )
    );
    nonflow_correlation_thread_sums = flow_correlation_thread_sums;
    combined_correlation_thread_sums = flow_correlation_thread_sums;
    flow_correlation_thread_counts = flow_correlation_thread_sums;
    nonflow_correlation_thread_counts = flow_correlation_thread_sums;
    combined_correlation_thread_counts = flow_correlation_thread_sums;

    if(stream_exponential_q) {
        flow_q_thread_bins.assign(
            static_cast<std::size_t>(ncore),
            std::vector<std::vector<double>>(
                static_cast<std::size_t>(Nsub),
                std::vector<double>(
                    static_cast<std::size_t>(flow_q_bin_layout.bins),
                    0.0
                )
            )
        );
        nonflow_q_thread_bins.assign(
            static_cast<std::size_t>(ncore),
            std::vector<std::vector<double>>(
                static_cast<std::size_t>(Nsub),
                std::vector<double>(
                    static_cast<std::size_t>(nonflow_q_bin_layout.bins),
                    0.0
                )
            )
        );
        combined_q_thread_bins.assign(
            static_cast<std::size_t>(ncore),
            std::vector<std::vector<double>>(
                static_cast<std::size_t>(Nsub),
                std::vector<double>(
                    static_cast<std::size_t>(combined_q_bin_layout.bins),
                    0.0
                )
            )
        );
    }

    if(need_raw_product_coefficients) {
        flow_product_thread_sums.assign(
            static_cast<std::size_t>(ncore),
            std::vector<std::vector<cd>>(
                static_cast<std::size_t>(Nsub),
                std::vector<cd>(static_cast<std::size_t>(M + 1), cd(0.0, 0.0))
            )
        );
        nonflow_product_thread_sums.assign(
            static_cast<std::size_t>(ncore),
            std::vector<std::vector<cd>>(
                static_cast<std::size_t>(Nsub),
                std::vector<cd>(
                    static_cast<std::size_t>(nonflow_particles + 1),
                    cd(0.0, 0.0)
                )
            )
        );
        combined_product_thread_sums.assign(
            static_cast<std::size_t>(ncore),
            std::vector<std::vector<cd>>(
                static_cast<std::size_t>(Nsub),
                std::vector<cd>(
                    static_cast<std::size_t>(event_multiplicity + 1),
                    cd(0.0, 0.0)
                )
            )
        );
        flow_product_thread_compensations = flow_product_thread_sums;
        nonflow_product_thread_compensations = nonflow_product_thread_sums;
        combined_product_thread_compensations = combined_product_thread_sums;
        product_thread_counts.assign(
            static_cast<std::size_t>(ncore),
            std::vector<double>(static_cast<std::size_t>(Nsub), 0.0)
        );
    }

    if(need_falling_subevent_coefficients) {
        flow_falling_subevent_thread_sums.assign(
            static_cast<std::size_t>(ncore),
            std::vector<std::vector<cd>>(
                static_cast<std::size_t>(Nsub),
                std::vector<cd>(static_cast<std::size_t>(M + 1), cd(0.0, 0.0))
            )
        );
        nonflow_falling_subevent_thread_sums.assign(
            static_cast<std::size_t>(ncore),
            std::vector<std::vector<cd>>(
                static_cast<std::size_t>(Nsub),
                std::vector<cd>(
                    static_cast<std::size_t>(nonflow_particles + 1),
                    cd(0.0, 0.0)
                )
            )
        );
        combined_falling_subevent_thread_sums.assign(
            static_cast<std::size_t>(ncore),
            std::vector<std::vector<cd>>(
                static_cast<std::size_t>(Nsub),
                std::vector<cd>(
                    static_cast<std::size_t>(event_multiplicity + 1),
                    cd(0.0, 0.0)
                )
            )
        );
        flow_falling_subevent_thread_compensations =
            flow_falling_subevent_thread_sums;
        nonflow_falling_subevent_thread_compensations =
            nonflow_falling_subevent_thread_sums;
        combined_falling_subevent_thread_compensations =
            combined_falling_subevent_thread_sums;
        falling_subevent_thread_counts.assign(
            static_cast<std::size_t>(ncore),
            std::vector<double>(static_cast<std::size_t>(Nsub), 0.0)
        );
    }

    if(use_modified_roots) {
        flow_modified_thread_sums.assign(
            static_cast<std::size_t>(ncore),
            std::vector<std::vector<cd>>(
                static_cast<std::size_t>(Nsub),
                std::vector<cd>(static_cast<std::size_t>(M + 1), cd(0.0, 0.0))
            )
        );
        nonflow_modified_thread_sums.assign(
            static_cast<std::size_t>(ncore),
            std::vector<std::vector<cd>>(
                static_cast<std::size_t>(Nsub),
                std::vector<cd>(
                    static_cast<std::size_t>(nonflow_particles + 1),
                    cd(0.0, 0.0)
                )
            )
        );
        combined_modified_thread_sums.assign(
            static_cast<std::size_t>(ncore),
            std::vector<std::vector<cd>>(
                static_cast<std::size_t>(Nsub),
                std::vector<cd>(
                    static_cast<std::size_t>(event_multiplicity + 1),
                    cd(0.0, 0.0)
                )
            )
        );
        flow_modified_thread_compensations = flow_modified_thread_sums;
        nonflow_modified_thread_compensations = nonflow_modified_thread_sums;
        combined_modified_thread_compensations = combined_modified_thread_sums;
        modified_thread_counts.assign(
            static_cast<std::size_t>(ncore),
            std::vector<double>(static_cast<std::size_t>(Nsub), 0.0)
        );
    }

    if(need_direct_vn_roots) {
        vn_thread_counts.assign(
            static_cast<std::size_t>(ncore),
            std::vector<std::vector<double>>(
                static_cast<std::size_t>(Nsub),
                std::vector<double>(static_cast<std::size_t>(vn_bins), 0.0)
            )
        );
    }

    omp_set_num_threads(ncore);
    #pragma omp parallel for schedule(dynamic)
    for(int i = 0; i < Nevents; ++i)
    {
        TRandom3 event_rng(cfg.seed + static_cast<unsigned int>(i));
        double psi2 = event_rng.Uniform(0.0, M_PI);
        double v2_evt = sample_v2(a, b, event_rng);
        const int tid = omp_get_thread_num();
        const std::size_t thread_index = static_cast<std::size_t>(tid);
        const std::size_t isub = static_cast<std::size_t>(i % Nsub);

        if(need_direct_vn_roots) {
            int ibin = static_cast<int>(
                v2_evt / b * static_cast<double>(vn_bins)
            );
            if(ibin < 0) ibin = 0;
            if(ibin >= vn_bins) ibin = vn_bins - 1;
            vn_thread_counts[thread_index][isub]
                [static_cast<std::size_t>(ibin)] += 1.0;
        }

        ToyEvent flow_event =
            has_flow
                ? generate_event(v2_evt, psi2, M, eta_max, event_rng)
                : ToyEvent();
        ToyEvent nonflow_event;
        AppendCopiedUniformNonflow(
            nonflow_event,
            nonflow_groups,
            nonflow_q,
            cfg.nonflow_delta,
            cfg.nonflow_delta_eta,
            eta_max,
            event_rng
        );
        ToyEvent combined_event = ConcatenateEvents(flow_event, nonflow_event);
        const EventCorrelations combined_event_correlations =
            ComputeEventCorrelations(n, combined_event.phi);
        EventCorrelations flow_event_correlations;
        EventCorrelations nonflow_event_correlations;
        if(has_flow) {
            flow_event_correlations = ComputeEventCorrelations(n, flow_event.phi);
            AddEventCorrelationsToSubsampleSums(
                flow_event_correlations,
                flow_correlation_thread_sums[thread_index][isub],
                flow_correlation_thread_counts[thread_index][isub]
            );
        }
        if(has_nonflow) {
            nonflow_event_correlations =
                ComputeEventCorrelations(n, nonflow_event.phi);
            AddEventCorrelationsToSubsampleSums(
                nonflow_event_correlations,
                nonflow_correlation_thread_sums[thread_index][isub],
                nonflow_correlation_thread_counts[thread_index][isub]
            );
        }
        AddEventCorrelationsToSubsampleSums(
            combined_event_correlations,
            combined_correlation_thread_sums[thread_index][isub],
            combined_correlation_thread_counts[thread_index][isub]
        );

        if(need_stored_correlations) {
            if(has_flow) {
                flow_correlations[static_cast<std::size_t>(i)] =
                    flow_event_correlations;
            }
            if(has_nonflow) {
                nonflow_correlations[static_cast<std::size_t>(i)] =
                    nonflow_event_correlations;
            }
            combined_correlations[static_cast<std::size_t>(i)] =
                combined_event_correlations;
        }

        if(use_exponential_roots) {
            const double q_nonflow =
                has_nonflow ? Qntheta(n, nonflow_event.phi, theta) : 0.0;
            const double q_combined = Qntheta(n, combined_event.phi, theta);
            if(stream_exponential_q) {
                if(has_flow) {
                    const double q_flow = Qntheta(n, flow_event.phi, theta);
                    const std::size_t flow_bin = static_cast<std::size_t>(
                        QBinIndex(q_flow, flow_q_bin_layout)
                    );
                    flow_q_thread_bins[thread_index][isub][flow_bin] += 1.0;
                }
                if(has_nonflow) {
                    const std::size_t nonflow_bin = static_cast<std::size_t>(
                        QBinIndex(q_nonflow, nonflow_q_bin_layout)
                    );
                    nonflow_q_thread_bins[thread_index][isub][nonflow_bin] += 1.0;
                }
                const std::size_t combined_bin = static_cast<std::size_t>(
                    QBinIndex(q_combined, combined_q_bin_layout)
                );
                combined_q_thread_bins[thread_index][isub][combined_bin] += 1.0;
            } else {
                if(has_flow) {
                    Q_flow[static_cast<std::size_t>(i)] =
                        Qntheta(n, flow_event.phi, theta);
                }
                if(has_nonflow) {
                    Q_nonflow[static_cast<std::size_t>(i)] = q_nonflow;
                }
                Q_combined[static_cast<std::size_t>(i)] = q_combined;
            }
        }

        if(need_raw_product_coefficients || use_modified_roots) {
            std::vector<cd> flow_product_coeffs;
            if(has_flow) {
                flow_product_coeffs =
                    BuildThetaAveragedProductCoefficients(
                        n,
                        theta,
                        cfg.product_theta_bins,
                        flow_event.phi
                    );
            }
            std::vector<cd> nonflow_product_coeffs;
            if(has_nonflow) {
                nonflow_product_coeffs =
                    BuildThetaAveragedProductCoefficients(
                        n,
                        theta,
                        cfg.product_theta_bins,
                        nonflow_event.phi
                    );
            }
            const std::vector<cd> combined_product_coeffs =
                BuildThetaAveragedProductCoefficients(
                    n,
                    theta,
                    cfg.product_theta_bins,
                    combined_event.phi
                );

            if(need_raw_product_coefficients) {
                if(has_flow) {
                    AddProductCoefficientsCompensated(
                        flow_product_thread_sums[thread_index][isub],
                        flow_product_thread_compensations[thread_index][isub],
                        flow_product_coeffs
                    );
                }
                if(has_nonflow) {
                    AddProductCoefficientsCompensated(
                        nonflow_product_thread_sums[thread_index][isub],
                        nonflow_product_thread_compensations[thread_index][isub],
                        nonflow_product_coeffs
                    );
                }
                AddProductCoefficientsCompensated(
                    combined_product_thread_sums[thread_index][isub],
                    combined_product_thread_compensations[thread_index][isub],
                    combined_product_coeffs
                );
                product_thread_counts[thread_index][isub] += 1.0;
            }

            if(use_modified_roots) {
                if(has_flow) {
                    AddProductCoefficientsCompensated(
                        flow_modified_thread_sums[thread_index][isub],
                        flow_modified_thread_compensations[thread_index][isub],
                        BuildModifiedProductCoefficients(
                            flow_product_coeffs,
                            flow_modified_factors
                        )
                    );
                }
                if(has_nonflow) {
                    AddProductCoefficientsCompensated(
                        nonflow_modified_thread_sums[thread_index][isub],
                        nonflow_modified_thread_compensations[thread_index][isub],
                        BuildModifiedProductCoefficients(
                            nonflow_product_coeffs,
                            nonflow_modified_factors
                        )
                    );
                }
                AddProductCoefficientsCompensated(
                    combined_modified_thread_sums[thread_index][isub],
                    combined_modified_thread_compensations[thread_index][isub],
                    BuildModifiedProductCoefficients(
                        combined_product_coeffs,
                        combined_modified_factors
                    )
                );
                modified_thread_counts[thread_index][isub] += 1.0;
            }

        }

        if(need_falling_subevent_coefficients) {
            if(has_flow) {
                const std::vector<cd> flow_coeffs =
                    BuildThetaAveragedEtaSubeventProductCoefficients(
                        n,
                        theta,
                        cfg.product_theta_bins,
                        flow_event,
                        eta_max,
                        cfg.Neta
                    );
                AddProductCoefficientsCompensated(
                    flow_falling_subevent_thread_sums[thread_index][isub],
                    flow_falling_subevent_thread_compensations[thread_index][isub],
                    flow_coeffs
                );
            }
            if(has_nonflow) {
                const std::vector<cd> nonflow_coeffs =
                    BuildThetaAveragedEtaSubeventProductCoefficients(
                        n,
                        theta,
                        cfg.product_theta_bins,
                        nonflow_event,
                        eta_max,
                        cfg.Neta
                    );
                AddProductCoefficientsCompensated(
                    nonflow_falling_subevent_thread_sums[thread_index][isub],
                    nonflow_falling_subevent_thread_compensations[thread_index][isub],
                    nonflow_coeffs
                );
            }
            const std::vector<cd> combined_coeffs =
                BuildThetaAveragedEtaSubeventProductCoefficients(
                    n,
                    theta,
                    cfg.product_theta_bins,
                    combined_event,
                    eta_max,
                    cfg.Neta
                );
            AddProductCoefficientsCompensated(
                combined_falling_subevent_thread_sums[thread_index][isub],
                combined_falling_subevent_thread_compensations[thread_index][isub],
                combined_coeffs
            );
            falling_subevent_thread_counts[thread_index][isub] += 1.0;
        }

        int done = 0;
        #pragma omp atomic capture
        done = ++completed_events;

        if(done % 10000 == 0 || done == Nevents) {
            #pragma omp critical(print_generation_progress)
            {
                std::cout << "Generated " << done << "/" << Nevents << " events\n";
            }
        }
    }

    std::cout << "Done\n";

    std::vector<std::vector<cd>> flow_product_subsample_sums;
    std::vector<std::vector<cd>> nonflow_product_subsample_sums;
    std::vector<std::vector<cd>> combined_product_subsample_sums;
    std::vector<std::vector<cd>> flow_product_subsample_compensations;
    std::vector<std::vector<cd>> nonflow_product_subsample_compensations;
    std::vector<std::vector<cd>> combined_product_subsample_compensations;
    std::vector<double> product_subsample_counts;
    std::vector<std::vector<cd>> flow_falling_subevent_subsample_sums;
    std::vector<std::vector<cd>> nonflow_falling_subevent_subsample_sums;
    std::vector<std::vector<cd>> combined_falling_subevent_subsample_sums;
    std::vector<std::vector<cd>> flow_falling_subevent_subsample_compensations;
    std::vector<std::vector<cd>> nonflow_falling_subevent_subsample_compensations;
    std::vector<std::vector<cd>> combined_falling_subevent_subsample_compensations;
    std::vector<double> falling_subevent_subsample_counts;
    std::vector<std::vector<cd>> flow_modified_subsample_sums;
    std::vector<std::vector<cd>> nonflow_modified_subsample_sums;
    std::vector<std::vector<cd>> combined_modified_subsample_sums;
    std::vector<std::vector<cd>> flow_modified_subsample_compensations;
    std::vector<std::vector<cd>> nonflow_modified_subsample_compensations;
    std::vector<std::vector<cd>> combined_modified_subsample_compensations;
    std::vector<double> modified_subsample_counts;
    std::vector<std::vector<double>> vn_subsample_counts;
    std::vector<std::vector<double>> flow_correlation_subsample_sums(
        static_cast<std::size_t>(Nsub),
        std::vector<double>(5, 0.0)
    );
    std::vector<std::vector<double>> combined_correlation_subsample_sums =
        flow_correlation_subsample_sums;
    std::vector<std::vector<double>> nonflow_correlation_subsample_sums =
        flow_correlation_subsample_sums;
    std::vector<std::vector<double>> flow_correlation_subsample_counts =
        flow_correlation_subsample_sums;
    std::vector<std::vector<double>> nonflow_correlation_subsample_counts =
        flow_correlation_subsample_sums;
    std::vector<std::vector<double>> combined_correlation_subsample_counts =
        flow_correlation_subsample_sums;
    std::vector<std::vector<double>> flow_q_subsample_bins;
    std::vector<std::vector<double>> nonflow_q_subsample_bins;
    std::vector<std::vector<double>> combined_q_subsample_bins;

    for(int tid = 0; tid < ncore; ++tid) {
        const std::size_t thread_index = static_cast<std::size_t>(tid);
        for(int isub = 0; isub < Nsub; ++isub) {
            const std::size_t sub_index = static_cast<std::size_t>(isub);
            for(int index = 0; index < 5; ++index) {
                const std::size_t corr_index = static_cast<std::size_t>(index);
                flow_correlation_subsample_sums[sub_index][corr_index] +=
                    flow_correlation_thread_sums[thread_index][sub_index][corr_index];
                combined_correlation_subsample_sums[sub_index][corr_index] +=
                    combined_correlation_thread_sums[thread_index][sub_index][corr_index];
                nonflow_correlation_subsample_sums[sub_index][corr_index] +=
                    nonflow_correlation_thread_sums[thread_index][sub_index][corr_index];
                flow_correlation_subsample_counts[sub_index][corr_index] +=
                    flow_correlation_thread_counts[thread_index][sub_index][corr_index];
                nonflow_correlation_subsample_counts[sub_index][corr_index] +=
                    nonflow_correlation_thread_counts[thread_index][sub_index][corr_index];
                combined_correlation_subsample_counts[sub_index][corr_index] +=
                    combined_correlation_thread_counts[thread_index][sub_index][corr_index];
            }
        }
    }

    if(stream_exponential_q) {
        flow_q_subsample_bins.assign(
            static_cast<std::size_t>(Nsub),
            std::vector<double>(
                static_cast<std::size_t>(flow_q_bin_layout.bins),
                0.0
            )
        );
        nonflow_q_subsample_bins.assign(
            static_cast<std::size_t>(Nsub),
            std::vector<double>(
                static_cast<std::size_t>(nonflow_q_bin_layout.bins),
                0.0
            )
        );
        combined_q_subsample_bins.assign(
            static_cast<std::size_t>(Nsub),
            std::vector<double>(
                static_cast<std::size_t>(combined_q_bin_layout.bins),
                0.0
            )
        );

        for(int tid = 0; tid < ncore; ++tid) {
            const std::size_t thread_index = static_cast<std::size_t>(tid);
            for(int isub = 0; isub < Nsub; ++isub) {
                const std::size_t sub_index = static_cast<std::size_t>(isub);
                for(int ibin = 0; ibin < flow_q_bin_layout.bins; ++ibin) {
                    const std::size_t bin_index = static_cast<std::size_t>(ibin);
                    flow_q_subsample_bins[sub_index][bin_index] +=
                        flow_q_thread_bins[thread_index][sub_index][bin_index];
                }
                for(int ibin = 0; ibin < combined_q_bin_layout.bins; ++ibin) {
                    const std::size_t bin_index = static_cast<std::size_t>(ibin);
                    combined_q_subsample_bins[sub_index][bin_index] +=
                        combined_q_thread_bins[thread_index][sub_index][bin_index];
                }
                for(int ibin = 0; ibin < nonflow_q_bin_layout.bins; ++ibin) {
                    const std::size_t bin_index = static_cast<std::size_t>(ibin);
                    nonflow_q_subsample_bins[sub_index][bin_index] +=
                        nonflow_q_thread_bins[thread_index][sub_index][bin_index];
                }
            }
        }
    }

    if(need_raw_product_coefficients) {
        flow_product_subsample_sums.assign(
            static_cast<std::size_t>(Nsub),
            std::vector<cd>(static_cast<std::size_t>(M + 1), cd(0.0, 0.0))
        );
        nonflow_product_subsample_sums.assign(
            static_cast<std::size_t>(Nsub),
            std::vector<cd>(
                static_cast<std::size_t>(nonflow_particles + 1),
                cd(0.0, 0.0)
            )
        );
        combined_product_subsample_sums.assign(
            static_cast<std::size_t>(Nsub),
            std::vector<cd>(
                    static_cast<std::size_t>(event_multiplicity + 1),
                    cd(0.0, 0.0)
                )
        );
        flow_product_subsample_compensations = flow_product_subsample_sums;
        nonflow_product_subsample_compensations = nonflow_product_subsample_sums;
        combined_product_subsample_compensations = combined_product_subsample_sums;
        product_subsample_counts.assign(static_cast<std::size_t>(Nsub), 0.0);

        for(int tid = 0; tid < ncore; ++tid) {
            const std::size_t thread_index = static_cast<std::size_t>(tid);

            for(int isub = 0; isub < Nsub; ++isub) {
                const std::size_t sub_index = static_cast<std::size_t>(isub);

                AddProductCoefficientsCompensated(
                    flow_product_subsample_sums[sub_index],
                    flow_product_subsample_compensations[sub_index],
                    flow_product_thread_sums[thread_index][sub_index]
                );
                AddProductCoefficientsCompensated(
                    combined_product_subsample_sums[sub_index],
                    combined_product_subsample_compensations[sub_index],
                    combined_product_thread_sums[thread_index][sub_index]
                );
                AddProductCoefficientsCompensated(
                    nonflow_product_subsample_sums[sub_index],
                    nonflow_product_subsample_compensations[sub_index],
                    nonflow_product_thread_sums[thread_index][sub_index]
                );

                product_subsample_counts[sub_index] +=
                    product_thread_counts[thread_index][sub_index];
            }
        }
    }

    if(need_falling_subevent_coefficients) {
        flow_falling_subevent_subsample_sums.assign(
            static_cast<std::size_t>(Nsub),
            std::vector<cd>(static_cast<std::size_t>(M + 1), cd(0.0, 0.0))
        );
        nonflow_falling_subevent_subsample_sums.assign(
            static_cast<std::size_t>(Nsub),
            std::vector<cd>(
                static_cast<std::size_t>(nonflow_particles + 1),
                cd(0.0, 0.0)
            )
        );
        combined_falling_subevent_subsample_sums.assign(
            static_cast<std::size_t>(Nsub),
            std::vector<cd>(
                static_cast<std::size_t>(event_multiplicity + 1),
                cd(0.0, 0.0)
            )
        );
        flow_falling_subevent_subsample_compensations =
            flow_falling_subevent_subsample_sums;
        nonflow_falling_subevent_subsample_compensations =
            nonflow_falling_subevent_subsample_sums;
        combined_falling_subevent_subsample_compensations =
            combined_falling_subevent_subsample_sums;
        falling_subevent_subsample_counts.assign(
            static_cast<std::size_t>(Nsub),
            0.0
        );

        for(int tid = 0; tid < ncore; ++tid) {
            const std::size_t thread_index = static_cast<std::size_t>(tid);

            for(int isub = 0; isub < Nsub; ++isub) {
                const std::size_t sub_index = static_cast<std::size_t>(isub);

                AddProductCoefficientsCompensated(
                    flow_falling_subevent_subsample_sums[sub_index],
                    flow_falling_subevent_subsample_compensations[sub_index],
                    flow_falling_subevent_thread_sums[thread_index][sub_index]
                );
                AddProductCoefficientsCompensated(
                    combined_falling_subevent_subsample_sums[sub_index],
                    combined_falling_subevent_subsample_compensations[sub_index],
                    combined_falling_subevent_thread_sums[thread_index][sub_index]
                );
                AddProductCoefficientsCompensated(
                    nonflow_falling_subevent_subsample_sums[sub_index],
                    nonflow_falling_subevent_subsample_compensations[sub_index],
                    nonflow_falling_subevent_thread_sums[thread_index][sub_index]
                );

                falling_subevent_subsample_counts[sub_index] +=
                    falling_subevent_thread_counts[thread_index][sub_index];
            }
        }
    }

    if(use_modified_roots) {
        flow_modified_subsample_sums.assign(
            static_cast<std::size_t>(Nsub),
            std::vector<cd>(static_cast<std::size_t>(M + 1), cd(0.0, 0.0))
        );
        nonflow_modified_subsample_sums.assign(
            static_cast<std::size_t>(Nsub),
            std::vector<cd>(
                static_cast<std::size_t>(nonflow_particles + 1),
                cd(0.0, 0.0)
            )
        );
        combined_modified_subsample_sums.assign(
            static_cast<std::size_t>(Nsub),
            std::vector<cd>(
                    static_cast<std::size_t>(event_multiplicity + 1),
                    cd(0.0, 0.0)
                )
        );
        flow_modified_subsample_compensations = flow_modified_subsample_sums;
        nonflow_modified_subsample_compensations = nonflow_modified_subsample_sums;
        combined_modified_subsample_compensations = combined_modified_subsample_sums;
        modified_subsample_counts.assign(static_cast<std::size_t>(Nsub), 0.0);

        for(int tid = 0; tid < ncore; ++tid) {
            const std::size_t thread_index = static_cast<std::size_t>(tid);
            for(int isub = 0; isub < Nsub; ++isub) {
                const std::size_t sub_index = static_cast<std::size_t>(isub);
                AddProductCoefficientsCompensated(
                    flow_modified_subsample_sums[sub_index],
                    flow_modified_subsample_compensations[sub_index],
                    flow_modified_thread_sums[thread_index][sub_index]
                );
                AddProductCoefficientsCompensated(
                    combined_modified_subsample_sums[sub_index],
                    combined_modified_subsample_compensations[sub_index],
                    combined_modified_thread_sums[thread_index][sub_index]
                );
                AddProductCoefficientsCompensated(
                    nonflow_modified_subsample_sums[sub_index],
                    nonflow_modified_subsample_compensations[sub_index],
                    nonflow_modified_thread_sums[thread_index][sub_index]
                );
                modified_subsample_counts[sub_index] +=
                    modified_thread_counts[thread_index][sub_index];
            }
        }
    }

    if(need_direct_vn_roots) {
        vn_subsample_counts.assign(
            static_cast<std::size_t>(Nsub),
            std::vector<double>(static_cast<std::size_t>(vn_bins), 0.0)
        );

        for(int tid = 0; tid < ncore; ++tid) {
            const std::size_t thread_index = static_cast<std::size_t>(tid);
            for(int isub = 0; isub < Nsub; ++isub) {
                const std::size_t sub_index = static_cast<std::size_t>(isub);
                for(int ibin = 0; ibin < vn_bins; ++ibin) {
                    const std::size_t bin_index =
                        static_cast<std::size_t>(ibin);
                    vn_subsample_counts[sub_index][bin_index] +=
                        vn_thread_counts[thread_index][sub_index][bin_index];
                }
            }
        }
    }

    if(has_flow && cfg.write_correlations) {
        std::cout << "Saving flow event correlations to "
                  << AddFilenameSuffix(cfg.correlations_output, "_flow")
                  << "\n" << std::flush;
        WriteEventCorrelations(
            AddFilenameSuffix(cfg.correlations_output, "_flow"),
            n,
            M,
            flow_correlations
        );
    } else {
        std::cout << "Skipping flow event correlation rows\n";
    }
    if(has_flow) {
        std::cout << "Saving flow correlation-derived G(z) coefficients to "
                  << AddFilenameSuffix(cfg.correlations_output, "_flow_coefficients")
                  << "\n" << std::flush;
        WriteCorrelationCoefficientEstimatesFromSubsamples(
            AddFilenameSuffix(cfg.correlations_output, "_flow_coefficients"),
            M,
            flow_correlation_subsample_sums,
            flow_correlation_subsample_counts
        );
    } else {
        std::cout << "Skipping flow correlation-derived G(z) coefficients; M_flow = 0\n";
    }
    if(has_nonflow && cfg.write_correlations) {
        std::cout << "Saving nonflow event correlations to "
                  << AddFilenameSuffix(cfg.correlations_output, "_nonflow")
                  << "\n" << std::flush;
        WriteEventCorrelations(
            AddFilenameSuffix(cfg.correlations_output, "_nonflow"),
            n,
            nonflow_particles,
            nonflow_correlations
        );
    } else {
        std::cout << "Skipping nonflow event correlation rows\n";
    }
    if(has_nonflow) {
        std::cout << "Saving nonflow correlation-derived G(z) coefficients to "
                  << AddFilenameSuffix(cfg.correlations_output, "_nonflow_coefficients")
                  << "\n" << std::flush;
        WriteCorrelationCoefficientEstimatesFromSubsamples(
            AddFilenameSuffix(cfg.correlations_output, "_nonflow_coefficients"),
            nonflow_particles,
            nonflow_correlation_subsample_sums,
            nonflow_correlation_subsample_counts
        );
    } else {
        std::cout << "Skipping nonflow correlation-derived G(z) coefficients; M_nonflow = 0\n";
    }
    if(cfg.write_correlations) {
        std::cout << "Saving combined event correlations to "
                  << AddFilenameSuffix(cfg.correlations_output, "_combined")
                  << "\n" << std::flush;
        WriteEventCorrelations(
            AddFilenameSuffix(cfg.correlations_output, "_combined"),
            n,
            event_multiplicity,
            combined_correlations
        );
    } else {
        std::cout << "Skipping combined event correlation rows\n";
    }
    std::cout << "Saving combined correlation-derived G(z) coefficients to "
              << AddFilenameSuffix(cfg.correlations_output, "_combined_coefficients")
              << "\n" << std::flush;
    WriteCorrelationCoefficientEstimatesFromSubsamples(
        AddFilenameSuffix(cfg.correlations_output, "_combined_coefficients"),
        event_multiplicity,
        combined_correlation_subsample_sums,
        combined_correlation_subsample_counts
    );
    std::cout << "Saved correlation-derived G(z) coefficients from "
              << cfg.correlations_output << "\n";

    if(cfg.write_correlation_truncated_lyz) {
        std::vector<std::vector<cd>> flow_corr_lyz_sums;
        std::vector<std::vector<cd>> nonflow_corr_lyz_sums;
        std::vector<std::vector<cd>> combined_corr_lyz_sums;
        std::vector<double> corr_lyz_counts;
        std::vector<double> nonflow_corr_lyz_counts;
        std::vector<double> combined_corr_lyz_counts;

        if(has_flow) {
            BuildCorrelationTruncatedSubsamples(
                flow_correlations,
                M,
                Nsub,
                flow_corr_lyz_sums,
                corr_lyz_counts
            );
        }
        BuildCorrelationTruncatedSubsamples(
            combined_correlations,
            event_multiplicity,
            Nsub,
            combined_corr_lyz_sums,
            combined_corr_lyz_counts
        );
        if(has_nonflow) {
            BuildCorrelationTruncatedSubsamples(
                nonflow_correlations,
                nonflow_particles,
                Nsub,
                nonflow_corr_lyz_sums,
                nonflow_corr_lyz_counts
            );
        }

        if(has_flow) {
            FindAndWriteProductRootsForDataset(
                "corr-lyz-flow",
                flow_corr_lyz_sums,
                corr_lyz_counts,
                no_coefficient_factors,
                false,
                Nresam,
                ncore,
                cfg.bootstrap_seed,
                cfg.root_start_points,
                AddFilenameSuffix(cfg.nominal_output, "_corr_lyz_flow"),
                AddFilenameSuffix(cfg.bootstrap_output, "_corr_lyz_flow")
            );
        }

        if(has_nonflow) {
            FindAndWriteProductRootsForDataset(
                "corr-lyz-nonflow",
                nonflow_corr_lyz_sums,
                nonflow_corr_lyz_counts,
                no_coefficient_factors,
                false,
                Nresam,
                ncore,
                cfg.bootstrap_seed,
                cfg.root_start_points,
                AddFilenameSuffix(cfg.nominal_output, "_corr_lyz_nonflow"),
                AddFilenameSuffix(cfg.bootstrap_output, "_corr_lyz_nonflow")
            );
        }

        FindAndWriteProductRootsForDataset(
            "corr-lyz-combined",
            combined_corr_lyz_sums,
            combined_corr_lyz_counts,
            no_coefficient_factors,
            false,
            Nresam,
            ncore,
            cfg.bootstrap_seed,
            cfg.root_start_points,
            AddFilenameSuffix(cfg.nominal_output, "_corr_lyz_combined"),
            AddFilenameSuffix(cfg.bootstrap_output, "_corr_lyz_combined")
        );
    }

    if(cfg.write_vn_lyz && has_flow) {
        FindAndWriteDirectVnLYZRoots(
            "vn-direct",
            vn_subsample_counts,
            b,
            static_cast<double>(M),
            M / 2,
            Nresam,
            ncore,
            cfg.bootstrap_seed,
            cfg.root_start_points,
            AddFilenameSuffix(cfg.nominal_output, "_vn_direct"),
            AddFilenameSuffix(cfg.bootstrap_output, "_vn_direct"),
            AddFilenameSuffix(cfg.vn_error_output, "_vn_direct"),
            cfg.vn_direct_error_source,
            cfg.coefficient_error_grid,
            cfg.coefficient_error_half_width,
            cfg.coefficient_error_re_min,
            cfg.coefficient_error_re_max,
            cfg.coefficient_error_im_min,
            cfg.coefficient_error_im_max
        );
    }

    if(cfg.write_vn_lyz && !has_flow)
        std::cout << "Skipping vn-direct roots; M_flow = 0\n";

    if(cfg.write_vn_j0 && has_flow) {
        std::vector<std::pair<double, double>> vn_j0_root_start_points;
        vn_j0_root_start_points.reserve(cfg.root_start_points.size());
        for(const auto& start : cfg.root_start_points) {
            vn_j0_root_start_points.push_back({
                static_cast<double>(M) * start.first,
                static_cast<double>(M) * start.second
            });
        }
        const double M_scale = static_cast<double>(M);
        const double vn_j0_error_half_width =
            cfg.coefficient_error_half_width > 0.0
                ? M_scale * cfg.coefficient_error_half_width
                : cfg.coefficient_error_half_width;
        const double vn_j0_error_re_min =
            std::isfinite(cfg.coefficient_error_re_min)
                ? M_scale * cfg.coefficient_error_re_min
                : cfg.coefficient_error_re_min;
        const double vn_j0_error_re_max =
            std::isfinite(cfg.coefficient_error_re_max)
                ? M_scale * cfg.coefficient_error_re_max
                : cfg.coefficient_error_re_max;
        const double vn_j0_error_im_min =
            std::isfinite(cfg.coefficient_error_im_min)
                ? M_scale * cfg.coefficient_error_im_min
                : cfg.coefficient_error_im_min;
        const double vn_j0_error_im_max =
            std::isfinite(cfg.coefficient_error_im_max)
                ? M_scale * cfg.coefficient_error_im_max
                : cfg.coefficient_error_im_max;

        FindAndWriteDirectVnLYZRoots(
            "vn-j0",
            vn_subsample_counts,
            b,
            1.0,
            -1,
            Nresam,
            ncore,
            cfg.bootstrap_seed,
            vn_j0_root_start_points,
            AddFilenameSuffix(cfg.nominal_output, "_vn_j0"),
            AddFilenameSuffix(cfg.bootstrap_output, "_vn_j0"),
            AddFilenameSuffix(cfg.vn_error_output, "_vn_j0"),
            cfg.vn_j0_error_source,
            cfg.coefficient_error_grid,
            vn_j0_error_half_width,
            vn_j0_error_re_min,
            vn_j0_error_re_max,
            vn_j0_error_im_min,
            vn_j0_error_im_max
        );
    }
    if(cfg.write_vn_j0 && !has_flow)
        std::cout << "Skipping vn-j0 roots; M_flow = 0\n";

    if(use_exponential_roots) {
        if(stream_exponential_q) {
            if(has_flow) {
                FindAndWriteRootsForBinnedDataset(
                    "flow",
                    flow_q_subsample_bins,
                    flow_q_bin_layout,
                    Nresam,
                    ncore,
                    cfg.bootstrap_seed,
                    cfg.root_start_points,
                    AddFilenameSuffix(cfg.nominal_output, "_flow"),
                    AddFilenameSuffix(cfg.bootstrap_output, "_flow")
                );
            }
            if(has_nonflow) {
                FindAndWriteRootsForBinnedDataset(
                    "nonflow",
                    nonflow_q_subsample_bins,
                    nonflow_q_bin_layout,
                    Nresam,
                    ncore,
                    cfg.bootstrap_seed,
                    cfg.root_start_points,
                    AddFilenameSuffix(cfg.nominal_output, "_nonflow"),
                    AddFilenameSuffix(cfg.bootstrap_output, "_nonflow")
                );
            }

            FindAndWriteRootsForBinnedDataset(
                "combined",
                combined_q_subsample_bins,
                combined_q_bin_layout,
                Nresam,
                ncore,
                cfg.bootstrap_seed,
                cfg.root_start_points,
                AddFilenameSuffix(cfg.nominal_output, "_combined"),
                AddFilenameSuffix(cfg.bootstrap_output, "_combined")
            );
        } else {
            if(has_flow) {
                FindAndWriteRootsForDataset(
                    "flow",
                    Q_flow,
                    M,
                    q_bins,
                    Nsub,
                    Nresam,
                    ncore,
                    cfg.bootstrap_seed,
                    cfg.root_start_points,
                    AddFilenameSuffix(cfg.nominal_output, "_flow"),
                    AddFilenameSuffix(cfg.bootstrap_output, "_flow")
                );
            }
            if(has_nonflow) {
                FindAndWriteRootsForDataset(
                    "nonflow",
                    Q_nonflow,
                    nonflow_particles,
                    q_bins,
                    Nsub,
                    Nresam,
                    ncore,
                    cfg.bootstrap_seed,
                    cfg.root_start_points,
                    AddFilenameSuffix(cfg.nominal_output, "_nonflow"),
                    AddFilenameSuffix(cfg.bootstrap_output, "_nonflow")
                );
            }

            FindAndWriteRootsForDataset(
                "combined",
                Q_combined,
                event_multiplicity,
                q_bins,
                Nsub,
                Nresam,
                ncore,
                cfg.bootstrap_seed,
                cfg.root_start_points,
                AddFilenameSuffix(cfg.nominal_output, "_combined"),
                AddFilenameSuffix(cfg.bootstrap_output, "_combined")
            );
        }
    }

    if(use_product_roots) {
        const std::string product_flow_suffix =
            use_exponential_roots ? "_product_flow" : "_flow";
        const std::string product_nonflow_suffix =
            use_exponential_roots ? "_product_nonflow" : "_nonflow";
        const std::string product_combined_suffix =
            use_exponential_roots ? "_product_combined" : "_combined";

        if(has_flow) {
            FindAndWriteProductRootsForDataset(
                "product-flow",
                flow_product_subsample_sums,
                product_subsample_counts,
                no_coefficient_factors,
                false,
                Nresam,
                ncore,
                cfg.bootstrap_seed,
                cfg.root_start_points,
                AddFilenameSuffix(cfg.nominal_output, product_flow_suffix),
                AddFilenameSuffix(cfg.bootstrap_output, product_flow_suffix),
                AddFilenameSuffix(cfg.root_diagnostics_output, product_flow_suffix),
                AddFilenameSuffix(cfg.coefficient_error_output, product_flow_suffix),
                cfg.coefficient_error_grid,
                cfg.coefficient_error_half_width,
                cfg.coefficient_error_re_min,
                cfg.coefficient_error_re_max,
                cfg.coefficient_error_im_min,
                cfg.coefficient_error_im_max
            );
        }

        if(has_nonflow) {
            FindAndWriteProductRootsForDataset(
                "product-nonflow",
                nonflow_product_subsample_sums,
                product_subsample_counts,
                no_coefficient_factors,
                false,
                Nresam,
                ncore,
                cfg.bootstrap_seed,
                cfg.root_start_points,
                AddFilenameSuffix(cfg.nominal_output, product_nonflow_suffix),
                AddFilenameSuffix(cfg.bootstrap_output, product_nonflow_suffix),
                AddFilenameSuffix(cfg.root_diagnostics_output, product_nonflow_suffix),
                AddFilenameSuffix(cfg.coefficient_error_output, product_nonflow_suffix),
                cfg.coefficient_error_grid,
                cfg.coefficient_error_half_width,
                cfg.coefficient_error_re_min,
                cfg.coefficient_error_re_max,
                cfg.coefficient_error_im_min,
                cfg.coefficient_error_im_max
            );
        }

        FindAndWriteProductRootsForDataset(
            "product-combined",
            combined_product_subsample_sums,
            product_subsample_counts,
            no_coefficient_factors,
            false,
            Nresam,
            ncore,
            cfg.bootstrap_seed,
            cfg.root_start_points,
            AddFilenameSuffix(cfg.nominal_output, product_combined_suffix),
            AddFilenameSuffix(cfg.bootstrap_output, product_combined_suffix),
            AddFilenameSuffix(cfg.root_diagnostics_output, product_combined_suffix),
            AddFilenameSuffix(cfg.coefficient_error_output, product_combined_suffix),
            cfg.coefficient_error_grid,
            cfg.coefficient_error_half_width,
            cfg.coefficient_error_re_min,
            cfg.coefficient_error_re_max,
            cfg.coefficient_error_im_min,
            cfg.coefficient_error_im_max
        );
    }

    if(use_product_falling_roots) {
        if(has_flow) {
            FindAndWriteProductRootsForDataset(
                "product-falling-flow",
                flow_product_subsample_sums,
                product_subsample_counts,
                flow_product_falling_factors,
                true,
                Nresam,
                ncore,
                cfg.bootstrap_seed,
                cfg.root_start_points,
                AddFilenameSuffix(cfg.nominal_output, "_product_falling_flow"),
                AddFilenameSuffix(cfg.bootstrap_output, "_product_falling_flow"),
                AddFilenameSuffix(cfg.root_diagnostics_output, "_product_falling_flow"),
                AddFilenameSuffix(cfg.coefficient_error_output, "_product_falling_flow"),
                cfg.coefficient_error_grid,
                cfg.coefficient_error_half_width,
                cfg.coefficient_error_re_min,
                cfg.coefficient_error_re_max,
                cfg.coefficient_error_im_min,
                cfg.coefficient_error_im_max
            );
        }

        if(has_nonflow) {
            FindAndWriteProductRootsForDataset(
                "product-falling-nonflow",
                nonflow_product_subsample_sums,
                product_subsample_counts,
                nonflow_product_falling_factors,
                true,
                Nresam,
                ncore,
                cfg.bootstrap_seed,
                cfg.root_start_points,
                AddFilenameSuffix(cfg.nominal_output, "_product_falling_nonflow"),
                AddFilenameSuffix(cfg.bootstrap_output, "_product_falling_nonflow"),
                AddFilenameSuffix(cfg.root_diagnostics_output, "_product_falling_nonflow"),
                AddFilenameSuffix(cfg.coefficient_error_output, "_product_falling_nonflow"),
                cfg.coefficient_error_grid,
                cfg.coefficient_error_half_width,
                cfg.coefficient_error_re_min,
                cfg.coefficient_error_re_max,
                cfg.coefficient_error_im_min,
                cfg.coefficient_error_im_max
            );
        }

        FindAndWriteProductRootsForDataset(
            "product-falling-combined",
            combined_product_subsample_sums,
            product_subsample_counts,
            combined_product_falling_factors,
            true,
            Nresam,
            ncore,
            cfg.bootstrap_seed,
            cfg.root_start_points,
            AddFilenameSuffix(cfg.nominal_output, "_product_falling_combined"),
            AddFilenameSuffix(cfg.bootstrap_output, "_product_falling_combined"),
            AddFilenameSuffix(cfg.root_diagnostics_output, "_product_falling_combined"),
            AddFilenameSuffix(cfg.coefficient_error_output, "_product_falling_combined"),
            cfg.coefficient_error_grid,
            cfg.coefficient_error_half_width,
            cfg.coefficient_error_re_min,
            cfg.coefficient_error_re_max,
            cfg.coefficient_error_im_min,
            cfg.coefficient_error_im_max
        );
    }

    if(use_product_falling_subevent_roots) {
        if(has_flow) {
            FindAndWriteProductRootsForDataset(
                "product-falling-subevents-flow",
                flow_falling_subevent_subsample_sums,
                falling_subevent_subsample_counts,
                no_coefficient_factors,
                true,
                Nresam,
                ncore,
                cfg.bootstrap_seed,
                cfg.root_start_points,
                AddFilenameSuffix(cfg.nominal_output, "_product_falling_subevents_flow"),
                AddFilenameSuffix(cfg.bootstrap_output, "_product_falling_subevents_flow"),
                AddFilenameSuffix(cfg.root_diagnostics_output, "_product_falling_subevents_flow"),
                AddFilenameSuffix(cfg.coefficient_error_output, "_product_falling_subevents_flow"),
                cfg.coefficient_error_grid,
                cfg.coefficient_error_half_width,
                cfg.coefficient_error_re_min,
                cfg.coefficient_error_re_max,
                cfg.coefficient_error_im_min,
                cfg.coefficient_error_im_max
            );
        }

        if(has_nonflow) {
            FindAndWriteProductRootsForDataset(
                "product-falling-subevents-nonflow",
                nonflow_falling_subevent_subsample_sums,
                falling_subevent_subsample_counts,
                no_coefficient_factors,
                true,
                Nresam,
                ncore,
                cfg.bootstrap_seed,
                cfg.root_start_points,
                AddFilenameSuffix(cfg.nominal_output, "_product_falling_subevents_nonflow"),
                AddFilenameSuffix(cfg.bootstrap_output, "_product_falling_subevents_nonflow"),
                AddFilenameSuffix(cfg.root_diagnostics_output, "_product_falling_subevents_nonflow"),
                AddFilenameSuffix(cfg.coefficient_error_output, "_product_falling_subevents_nonflow"),
                cfg.coefficient_error_grid,
                cfg.coefficient_error_half_width,
                cfg.coefficient_error_re_min,
                cfg.coefficient_error_re_max,
                cfg.coefficient_error_im_min,
                cfg.coefficient_error_im_max
            );
        }

        FindAndWriteProductRootsForDataset(
            "product-falling-subevents-combined",
            combined_falling_subevent_subsample_sums,
            falling_subevent_subsample_counts,
            no_coefficient_factors,
            true,
            Nresam,
            ncore,
            cfg.bootstrap_seed,
            cfg.root_start_points,
            AddFilenameSuffix(cfg.nominal_output, "_product_falling_subevents_combined"),
            AddFilenameSuffix(cfg.bootstrap_output, "_product_falling_subevents_combined"),
            AddFilenameSuffix(cfg.root_diagnostics_output, "_product_falling_subevents_combined"),
            AddFilenameSuffix(cfg.coefficient_error_output, "_product_falling_subevents_combined"),
            cfg.coefficient_error_grid,
            cfg.coefficient_error_half_width,
            cfg.coefficient_error_re_min,
            cfg.coefficient_error_re_max,
            cfg.coefficient_error_im_min,
            cfg.coefficient_error_im_max
        );
    }

    if(use_product_correction_roots) {
        if(has_flow) {
            FindAndWriteProductRootsForDataset(
                "product-correction-flow",
                flow_product_subsample_sums,
                product_subsample_counts,
                flow_product_correction_factors,
                false,
                Nresam,
                ncore,
                cfg.bootstrap_seed,
                cfg.root_start_points,
                AddFilenameSuffix(cfg.nominal_output, "_product_correction_flow"),
                AddFilenameSuffix(cfg.bootstrap_output, "_product_correction_flow"),
                AddFilenameSuffix(cfg.root_diagnostics_output, "_product_correction_flow"),
                AddFilenameSuffix(cfg.coefficient_error_output, "_product_correction_flow"),
                cfg.coefficient_error_grid,
                cfg.coefficient_error_half_width,
                cfg.coefficient_error_re_min,
                cfg.coefficient_error_re_max,
                cfg.coefficient_error_im_min,
                cfg.coefficient_error_im_max
            );
        }

        if(has_nonflow) {
            FindAndWriteProductRootsForDataset(
                "product-correction-nonflow",
                nonflow_product_subsample_sums,
                product_subsample_counts,
                nonflow_product_correction_factors,
                false,
                Nresam,
                ncore,
                cfg.bootstrap_seed,
                cfg.root_start_points,
                AddFilenameSuffix(cfg.nominal_output, "_product_correction_nonflow"),
                AddFilenameSuffix(cfg.bootstrap_output, "_product_correction_nonflow"),
                AddFilenameSuffix(cfg.root_diagnostics_output, "_product_correction_nonflow"),
                AddFilenameSuffix(cfg.coefficient_error_output, "_product_correction_nonflow"),
                cfg.coefficient_error_grid,
                cfg.coefficient_error_half_width,
                cfg.coefficient_error_re_min,
                cfg.coefficient_error_re_max,
                cfg.coefficient_error_im_min,
                cfg.coefficient_error_im_max
            );
        }

        FindAndWriteProductRootsForDataset(
            "product-correction-combined",
            combined_product_subsample_sums,
            product_subsample_counts,
            combined_product_correction_factors,
            false,
            Nresam,
            ncore,
            cfg.bootstrap_seed,
            cfg.root_start_points,
            AddFilenameSuffix(cfg.nominal_output, "_product_correction_combined"),
            AddFilenameSuffix(cfg.bootstrap_output, "_product_correction_combined"),
            AddFilenameSuffix(cfg.root_diagnostics_output, "_product_correction_combined"),
            AddFilenameSuffix(cfg.coefficient_error_output, "_product_correction_combined"),
            cfg.coefficient_error_grid,
            cfg.coefficient_error_half_width,
            cfg.coefficient_error_re_min,
            cfg.coefficient_error_re_max,
            cfg.coefficient_error_im_min,
            cfg.coefficient_error_im_max
        );
    }

    if(use_product_correction2_roots) {
        if(has_flow) {
            FindAndWriteProductRootsForDataset(
                "product-correction2-flow",
                flow_product_subsample_sums,
                product_subsample_counts,
                flow_product_correction2_factors,
                false,
                Nresam,
                ncore,
                cfg.bootstrap_seed,
                cfg.root_start_points,
                AddFilenameSuffix(cfg.nominal_output, "_product_correction2_flow"),
                AddFilenameSuffix(cfg.bootstrap_output, "_product_correction2_flow"),
                AddFilenameSuffix(cfg.root_diagnostics_output, "_product_correction2_flow"),
                AddFilenameSuffix(cfg.coefficient_error_output, "_product_correction2_flow"),
                cfg.coefficient_error_grid,
                cfg.coefficient_error_half_width,
                cfg.coefficient_error_re_min,
                cfg.coefficient_error_re_max,
                cfg.coefficient_error_im_min,
                cfg.coefficient_error_im_max
            );
        }

        if(has_nonflow) {
            FindAndWriteProductRootsForDataset(
                "product-correction2-nonflow",
                nonflow_product_subsample_sums,
                product_subsample_counts,
                nonflow_product_correction2_factors,
                false,
                Nresam,
                ncore,
                cfg.bootstrap_seed,
                cfg.root_start_points,
                AddFilenameSuffix(cfg.nominal_output, "_product_correction2_nonflow"),
                AddFilenameSuffix(cfg.bootstrap_output, "_product_correction2_nonflow"),
                AddFilenameSuffix(cfg.root_diagnostics_output, "_product_correction2_nonflow"),
                AddFilenameSuffix(cfg.coefficient_error_output, "_product_correction2_nonflow"),
                cfg.coefficient_error_grid,
                cfg.coefficient_error_half_width,
                cfg.coefficient_error_re_min,
                cfg.coefficient_error_re_max,
                cfg.coefficient_error_im_min,
                cfg.coefficient_error_im_max
            );
        }

        FindAndWriteProductRootsForDataset(
            "product-correction2-combined",
            combined_product_subsample_sums,
            product_subsample_counts,
            combined_product_correction2_factors,
            false,
            Nresam,
            ncore,
            cfg.bootstrap_seed,
            cfg.root_start_points,
            AddFilenameSuffix(cfg.nominal_output, "_product_correction2_combined"),
            AddFilenameSuffix(cfg.bootstrap_output, "_product_correction2_combined"),
            AddFilenameSuffix(cfg.root_diagnostics_output, "_product_correction2_combined"),
            AddFilenameSuffix(cfg.coefficient_error_output, "_product_correction2_combined"),
            cfg.coefficient_error_grid,
            cfg.coefficient_error_half_width,
            cfg.coefficient_error_re_min,
            cfg.coefficient_error_re_max,
            cfg.coefficient_error_im_min,
            cfg.coefficient_error_im_max
        );
    }

    if(use_modified_roots) {
        if(has_flow) {
            FindAndWriteProductRootsForDataset(
                "modified-flow",
                flow_modified_subsample_sums,
                modified_subsample_counts,
                no_coefficient_factors,
                false,
                Nresam,
                ncore,
                cfg.bootstrap_seed,
                cfg.root_start_points,
                AddFilenameSuffix(cfg.nominal_output, "_modified_flow"),
                AddFilenameSuffix(cfg.bootstrap_output, "_modified_flow"),
                AddFilenameSuffix(cfg.root_diagnostics_output, "_modified_flow"),
                AddFilenameSuffix(cfg.coefficient_error_output, "_modified_flow"),
                cfg.coefficient_error_grid,
                cfg.coefficient_error_half_width,
                cfg.coefficient_error_re_min,
                cfg.coefficient_error_re_max,
                cfg.coefficient_error_im_min,
                cfg.coefficient_error_im_max
            );
        }

        if(has_nonflow) {
            FindAndWriteProductRootsForDataset(
                "modified-nonflow",
                nonflow_modified_subsample_sums,
                modified_subsample_counts,
                no_coefficient_factors,
                false,
                Nresam,
                ncore,
                cfg.bootstrap_seed,
                cfg.root_start_points,
                AddFilenameSuffix(cfg.nominal_output, "_modified_nonflow"),
                AddFilenameSuffix(cfg.bootstrap_output, "_modified_nonflow"),
                AddFilenameSuffix(cfg.root_diagnostics_output, "_modified_nonflow"),
                AddFilenameSuffix(cfg.coefficient_error_output, "_modified_nonflow"),
                cfg.coefficient_error_grid,
                cfg.coefficient_error_half_width,
                cfg.coefficient_error_re_min,
                cfg.coefficient_error_re_max,
                cfg.coefficient_error_im_min,
                cfg.coefficient_error_im_max
            );
        }

        FindAndWriteProductRootsForDataset(
            "modified-combined",
            combined_modified_subsample_sums,
            modified_subsample_counts,
            no_coefficient_factors,
            false,
            Nresam,
            ncore,
            cfg.bootstrap_seed,
            cfg.root_start_points,
            AddFilenameSuffix(cfg.nominal_output, "_modified_combined"),
            AddFilenameSuffix(cfg.bootstrap_output, "_modified_combined"),
            AddFilenameSuffix(cfg.root_diagnostics_output, "_modified_combined"),
            AddFilenameSuffix(cfg.coefficient_error_output, "_modified_combined"),
            cfg.coefficient_error_grid,
            cfg.coefficient_error_half_width,
            cfg.coefficient_error_re_min,
            cfg.coefficient_error_re_max,
            cfg.coefficient_error_im_min,
            cfg.coefficient_error_im_max
        );
    }

    if(use_exact_roots) {
        if(has_flow) {
            FindAndWriteProductRootsForDataset(
                "exact-flow",
                flow_product_subsample_sums,
                product_subsample_counts,
                flow_exact_factors,
                true,
                Nresam,
                ncore,
                cfg.bootstrap_seed,
                cfg.root_start_points,
                AddFilenameSuffix(cfg.nominal_output, "_exact_flow"),
                AddFilenameSuffix(cfg.bootstrap_output, "_exact_flow"),
                AddFilenameSuffix(cfg.root_diagnostics_output, "_exact_flow"),
                AddFilenameSuffix(cfg.coefficient_error_output, "_exact_flow"),
                cfg.coefficient_error_grid,
                cfg.coefficient_error_half_width,
                cfg.coefficient_error_re_min,
                cfg.coefficient_error_re_max,
                cfg.coefficient_error_im_min,
                cfg.coefficient_error_im_max
            );
        }

        if(has_nonflow) {
            FindAndWriteProductRootsForDataset(
                "exact-nonflow",
                nonflow_product_subsample_sums,
                product_subsample_counts,
                nonflow_exact_factors,
                true,
                Nresam,
                ncore,
                cfg.bootstrap_seed,
                cfg.root_start_points,
                AddFilenameSuffix(cfg.nominal_output, "_exact_nonflow"),
                AddFilenameSuffix(cfg.bootstrap_output, "_exact_nonflow"),
                AddFilenameSuffix(cfg.root_diagnostics_output, "_exact_nonflow"),
                AddFilenameSuffix(cfg.coefficient_error_output, "_exact_nonflow"),
                cfg.coefficient_error_grid,
                cfg.coefficient_error_half_width,
                cfg.coefficient_error_re_min,
                cfg.coefficient_error_re_max,
                cfg.coefficient_error_im_min,
                cfg.coefficient_error_im_max
            );
        }

        FindAndWriteProductRootsForDataset(
            "exact-combined",
            combined_product_subsample_sums,
            product_subsample_counts,
            combined_exact_factors,
            true,
            Nresam,
            ncore,
            cfg.bootstrap_seed,
            cfg.root_start_points,
            AddFilenameSuffix(cfg.nominal_output, "_exact_combined"),
            AddFilenameSuffix(cfg.bootstrap_output, "_exact_combined"),
            AddFilenameSuffix(cfg.root_diagnostics_output, "_exact_combined"),
            AddFilenameSuffix(cfg.coefficient_error_output, "_exact_combined"),
            cfg.coefficient_error_grid,
            cfg.coefficient_error_half_width,
            cfg.coefficient_error_re_min,
            cfg.coefficient_error_re_max,
            cfg.coefficient_error_im_min,
            cfg.coefficient_error_im_max
        );
    }

    /*
    // Plotting code kept for later.
    const int Nr = 200;
    const double r_min = 0.0;
    const double r_max = 0.4;
    const double imag_shift = 0.05420014849053053;

    std::vector<double> r_values(Nr);
    std::vector<double> G_abs(Nr);
    std::vector<double> G_real(Nr);
    std::vector<double> G_imag(Nr);
    std::vector<double> G_triangle(Nr);

    std::vector<double> G_triangle_real_complex(Nr);
    std::vector<double> G_triangle_imag_complex(Nr);

    for(int ir = 0; ir < Nr; ++ir)
    {
        double r = r_min + (r_max - r_min) * ir / double(Nr - 1);
        cd z(r, imag_shift);

        r_values[ir] = r;

        cd sumG = 0.0;
        cd sumGz = 0.0;

        for(int iev = 0; iev < Nevents; ++iev)
        {
            sumG  += std::exp(cd(0.0, r * Q[iev]));
            sumGz += std::exp(cd(0.0, 1.0) * z * Q[iev]);
        }

        cd G  = sumG  / double(Nevents);
        cd Gz = sumGz / double(Nevents);

        G_abs[ir]  = std::abs(G);
        G_real[ir] = std::real(Gz);
        G_imag[ir] = std::imag(Gz);

        cd tri = std::exp(-double(M) * r * r / 4.0) * triangleG(double(M) * r, a, b);
        G_triangle[ir] = std::abs(tri);

        cd tri_complex = std::exp(-double(M) * z * z / 4.0) * triangleG(double(M) * z, a, b);
        G_triangle_real_complex[ir] = std::real(tri_complex);
        G_triangle_imag_complex[ir] = std::imag(tri_complex);

        if((ir + 1) % 10 == 0)
            std::cout << ir + 1 << "/" << Nr << "\n";
    }

    TGraph* g = new TGraph(Nr);
    TGraph* g_tri = new TGraph(Nr);

    for(int i = 0; i < Nr; ++i)
    {
        g->SetPoint(i, r_values[i], G_abs[i]);
        g_tri->SetPoint(i, r_values[i], G_triangle[i]);
    }

    TCanvas* c = new TCanvas("c_Gabs", "", 800, 600);

    g->SetTitle(";r;|<e^{i r Q}>|");
    g->SetLineWidth(3);
    g->Draw("AL");

    g->GetXaxis()->SetRangeUser(0.0, 0.35);
    g->GetYaxis()->SetRangeUser(0.0, 0.1);

    g_tri->SetLineColor(kRed);
    g_tri->SetLineWidth(3);
    g_tri->SetLineStyle(2);
    g_tri->Draw("L same");

    TLegend* leg = new TLegend(0.55, 0.70, 0.88, 0.88);
    leg->AddEntry(g, "Monte Carlo", "l");
    leg->AddEntry(g_tri, "triangleG", "l");
    leg->Draw();

    c->SaveAs("Gabs_triangleG_ROOT.pdf");
    c->SaveAs("Gabs_triangleG_ROOT.png");
    c->Draw();

    TGraph* g_real = new TGraph(Nr);
    TGraph* g_imag = new TGraph(Nr);
    TGraph* g_tri_real = new TGraph(Nr);
    TGraph* g_tri_imag = new TGraph(Nr);

    for(int i = 0; i < Nr; ++i)
    {
        g_real->SetPoint(i, r_values[i], G_real[i]);
        g_imag->SetPoint(i, r_values[i], G_imag[i]);
        g_tri_real->SetPoint(i, r_values[i], G_triangle_real_complex[i]);
        g_tri_imag->SetPoint(i, r_values[i], G_triangle_imag_complex[i]);
    }

    g_real->SetLineColor(kBlue);
    g_real->SetLineWidth(3);
    g_real->SetLineStyle(1);

    g_imag->SetLineColor(kRed);
    g_imag->SetLineWidth(3);
    g_imag->SetLineStyle(1);

    g_tri_real->SetLineColor(kBlue);
    g_tri_real->SetLineWidth(3);
    g_tri_real->SetLineStyle(2);

    g_tri_imag->SetLineColor(kRed);
    g_tri_imag->SetLineWidth(3);
    g_tri_imag->SetLineStyle(2);

    TCanvas* c_compare = new TCanvas("c_compare", "", 900, 700);

    g_real->SetTitle(";r;G(r+i0.0542)");
    g_real->Draw("AL");

    g_real->GetXaxis()->SetRangeUser(0.1, 0.2);
    g_real->GetYaxis()->SetRangeUser(-0.01, 0.01);

    g_imag->Draw("L same");
    g_tri_real->Draw("L same");
    g_tri_imag->Draw("L same");

    TLegend* leg_compare = new TLegend(0.15, 0.75, 0.30, 0.90);
    leg_compare->AddEntry(g_real, "Re G", "l");
    leg_compare->AddEntry(g_imag, "Im G", "l");
    leg_compare->AddEntry(g_tri_real, "Re triangleG", "l");
    leg_compare->AddEntry(g_tri_imag, "Im triangleG", "l");
    leg_compare->Draw();

    c_compare->SaveAs("G_complex_triangleG_ROOT.pdf");
    c_compare->SaveAs("G_complex_triangleG_ROOT.png");
    c_compare->Draw();
    */

    return 0;
}

// ------------------------------------------------------------
// ROOT macro wrapper.
// Run with:
// root -l triangleG.C
// ------------------------------------------------------------
void triangleG()
{
    TriangleGConfig cfg;
    ApplyOutputFolder(cfg);
    RunTriangleG(cfg);
}

#ifndef __CLING__
int main(int argc, char** argv)
{
    try {
        TriangleGConfig cfg = ParseTriangleGArgs(argc, argv);
        ApplyOutputFolder(cfg);
        return RunTriangleG(cfg);
    }
    catch(const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        std::cerr << "Run with --help for usage.\n";
        return 1;
    }
}
#endif
