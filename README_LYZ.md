# LYZ project build notes

This project is now built as a normal C++ executable instead of a ROOT macro.
That is important because the complex Bessel library uses C++ plus Fortran/AMOS,
which is much cleaner to link with CMake than through ROOT/Cling.

## Build and run

```bash
./run.sh PbPb_central_4.dat
```

or, manually:

```bash
mkdir -p build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
./build/LYZ PbPb_central_4.dat
```

## Complex Bessel function

The complex Bessel function is now computed with

```cpp
sp_bessel::besselJ(0.0, z, false, &error)
```

from the bundled `complex_bessel` library, which wraps AMOS.
This replaces the failing Boost call.
