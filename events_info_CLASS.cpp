#include "TObject.h"
#include "events_info_CLASS.h"

#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
using namespace std;

ClassImp(events_info_CLASS)

events_info_CLASS::events_info_CLASS() {  };

bool events_info_CLASS::check_trento_opt(TString s){
    bool check = false;
    for (pair<TString, TString> element : options){
        if (s == element.first) check = true;
    } 
    if (!check) {
        cout << "ERROR: TrENTo option " + s + " is not found." << endl;
    }    
    return check;
}

void events_info_CLASS::Set(TString s1, TString s2){
    if (check_trento_opt(s1)){
        if (s1 == "target" || s1 == "projectile"){ 
        options[s1] = s2;
        }
        if (s2 == "") return; /* This line is added because of the following: In the bash script, if the variable is empty then I get Set(var_name, )
        and Root (c++) does not let it even if the function is not called. So I should return an empty string to get rid of the problem. */      
    }        
};
void events_info_CLASS::Set(TString s1, Int_t s2){
    if (check_trento_opt(s1)){
        if (s1 == "nevents"){
            options[s1] = Form("%d", (int)s2);
        } else if (s1 == "constit-number" || s1 == "m"){
            options["constit-number"] = Form("%d", (int)s2);
        } else if (s1 == "black-disk-nucleon" || s1 == "g"){
            options["black-disk-nucleon"] = Form("%d", (int)s2);
        } else if (s1 == "long-output" || s1 == "l"){
            if (s2 == 0 || s2 == 1){
                options["long-output"] = Form("%d", (int)s2);
            } else {
                cout << "WARRNING: The input for \"" << s1 << "\" should be a boolean (0 or 1)." << endl;
            }
        } else {
            cout << "WARRNING: The input for \"" << s1 << "\" should be a fload (double), but it is OK." << endl;
            double doub_s2 = s2;
            Set(s1, doub_s2);  
        }       
    }    
};
void events_info_CLASS::Set(TString s1, Double_t s2){
    if (check_trento_opt(s1)){
        if (s1 == "reduced-thickness" || s1 == "p") {
            options["reduced-thickness"] = Form("%g", (double)s2);
        } else if (s1 == "reduced-thickness-extra" || s1 == "q") {
            options["reduced-thickness-extra"] = Form("%g", (double)s2);
        } else if (s1 == "Eref" || s1 == "e") {
            options["Eref"] = Form("%g", (double)s2);
        } else if (s1 == "fluctuation" || s1 == "k"){
            options["fluctuation"] = Form("%g", (double)s2);
        } else if (s1 == "nucleon-width" || s1 == "w"){
            options["nucleon-width"] = Form("%g", (double)s2);
        } else if (s1 == "constit-width" || s1 == "v"){
            options["constit-width"] = Form("%g", (double)s2);
        } else if (s1 == "nucleon-min-dist" || s1 == "d"){
            options["nucleon-min-dist"] = Form("%g", (double)s2);
        } else if (s1 == "cross-section" || s1 == "x"){
            options["cross-section"] = Form("%g", (double)s2);
        } else if (s1 == "normalization" || s1 == "n"){
            options["normalization"] = Form("%g", (double)s2);
        } else if ( s1 == "b-min" || s1 == "b-max" || s1 == "grid-max" || s1 == "grid-step"){
            options[s1] = Form("%g", (double)s2);
        } else {
            cout << "WARRNING: The input for \"" << s1 << "\" should be an integer, digits after decimal point are dropped." << endl;
            int int_s2 = s2;
            Set(s1, int_s2);                  
        }  
    }     
};

void events_info_CLASS::Set(map<TString, TString> input){
    for (pair<TString, TString> element : input){
        options[element.first] = element.second;
    }     
} 
void events_info_CLASS::Set_from_config_file(string s){

    if (s == "") return;

    string line;
    ifstream config_file(s);

    stringstream split_line;
    istream_iterator<string> begin, end;
    vector<string> vector_line;
    if (config_file.is_open()){
        while (getline(config_file, line)){
            if (line.front() == '#' || line.length() == 0) continue;
            split_line = stringstream(line);
            begin = istream_iterator<string>(split_line);
            vector_line = vector<string>(begin, end);
            if (vector_line.size() == 2 || (vector_line.size() > 2 && vector_line[2].front() == '#')){
                bool check_list = false;
                for (pair<TString, TString> element: options){
                    if (vector_line[0] == element.first){
                        check_list = true;
                    }
                }
                if (!check_list){
                    cout << "ERROR: TrENTo option " + vector_line[0] + " is not found." << endl;
                    return;
                }
                options[vector_line[0]] = vector_line[1];
            } else {
                cout << "ERROR: Each line of config file should have two items: 'option name' and 'option value'." << endl;
                break;
            };
        }
    }
}
void events_info_CLASS::Set_from_root_file(TString s){
    unique_ptr<TFile> f(TFile::Open(s,"READ"));
    events_info_CLASS *read_info;
    f->GetObject("events_information", read_info);
    for (pair<TString, TString> element: options){
        options[element.first] = read_info->Get()[element.first];
    }
}
void events_info_CLASS::Save_to_root_file(TString s, TString file_option){
    unique_ptr<TFile> f(TFile::Open(s,file_option));
    events_info_CLASS write_info;
    write_info.Set(options);
    f->WriteObject(&write_info, "events_information");
}
void events_info_CLASS::Save_to_config_file(string s){
    ofstream out_file;
    out_file.open(s);
    out_file << "# TrENTo configuration:\n";
    out_file << "# ====================\n\n";
    for (pair<TString, TString> element: options){
        if (element.second == "") continue;
        out_file << element.first << "\t" << element.second << endl;
    }
}
map<TString, TString> events_info_CLASS::Get(){
    map<TString, TString> result_map;
    for (pair<TString, TString> element : options){
        result_map[element.first] = element.second;
        // cout << "What is happening: " << element.first << "    " << element.second << endl;
    }
    return result_map;        
}