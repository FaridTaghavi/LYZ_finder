void LYZ(const char* filename = "PbPb_central_4.dat");

int main(int argc, char** argv)
{
    const char* filename = (argc > 1) ? argv[1] : "PbPb_central_4.dat";
    LYZ(filename);
    return 0;
}
