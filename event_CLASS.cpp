#include "TObject.h"
#include "event_CLASS.h"

ClassImp(event_CLASS)

event_CLASS::event_CLASS() { }

void event_CLASS::Set_epsilon(Int_t n, Double_t ex, Double_t ey)
{
    epsilonx[n] = ex; epsilony[n] = ey;
}
void event_CLASS::Set_r(Int_t n, Double_t ar)
{
    r[n] = ar;
}
void event_CLASS::Set_npart(Double_t np)
{
    npart = np;
}
void event_CLASS::Set_nch(Double_t nc)
{
    nch = nc;
}
void event_CLASS::Set_b(Double_t impact_p)
{
    b = impact_p;
}

Double_t event_CLASS::Get_epsilonx(Int_t n) {return epsilonx[n];}
Double_t event_CLASS::Get_epsilony(Int_t n) {return epsilony[n];}
Double_t event_CLASS::Get_r(Int_t n) {return r[n];}
Double_t event_CLASS::Get_npart() {return npart;}
Double_t event_CLASS::Get_nch() {return nch;}
Double_t event_CLASS::Get_b() {return b;}