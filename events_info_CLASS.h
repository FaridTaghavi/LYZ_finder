#ifndef events_info_CLASS_h
#define events_info_CLASS_h

#include "TObject.h"
#include "TROOT.h" 
#include "TFile.h" 

//#include "trento_options.h"

#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
using namespace std;

class events_info_CLASS: public TObject
{
private:
    map<TString, TString> options = {{"target", "Pb"}, {"projectile", "Pb"} , {"nevents" , "1"}, {"reduced-thickness", "0.0"}, 
    {"fluctuation", "1.0"}, {"nucleon-width", "0.5"}, {"constit-width", ""}, {"constit-number", "1"}, {"nucleon-min-dist", "0.0"},
    {"cross-section", "6.4"}, {"normalization", "1.0"}, {"b-min", ""}, {"b-max", ""}, {"grid-max", "10.0"}, 
    {"grid-step", "0.2"}, {"long-output", "0"}, {"black-disk-nucleon", "0"}, {"reduced-thickness-extra", "1"}, {"Eref", "1"}};
public:
    events_info_CLASS();

    bool check_trento_opt(TString);
    void Set(TString, TString);
    void Set(TString, Int_t);
    void Set(TString, Double_t);
    void Set(map<TString, TString>);
    void Set_from_config_file(string);
    void Set_from_root_file(TString);
    void Save_to_root_file(TString, TString);
    void Save_to_config_file(string);
    map<TString, TString> Get();

    ClassDef(events_info_CLASS, 1)
};

#endif