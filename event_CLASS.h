#ifndef event_CLASS_h
#define event_CLASS_h


#include "TObject.h"

//#include <iostream>

//using namespace std;

class event_CLASS: public TObject {

private:
    // TString info_lable[10];
    // Double_t info_entry[10] = {0};
    //===============
    Double_t epsilonx[9] = {0};
    Double_t epsilony[9] = {0};
    Double_t r[9] = {0};
    Double_t npart = 0;
    Double_t nch = 0;
    Double_t b = 0;


public:

    event_CLASS();

    void Set_epsilon(Int_t, Double_t, Double_t);
    void Set_r(Int_t, Double_t);
    void Set_npart(Double_t);
    void Set_nch(Double_t);
    void Set_b(Double_t);

    Double_t Get_epsilonx(Int_t);
    Double_t Get_epsilony(Int_t);
    Double_t Get_r(Int_t);
    Double_t Get_npart();
    Double_t Get_nch();
    Double_t Get_b();

    ClassDef(event_CLASS, 1)
};


#endif