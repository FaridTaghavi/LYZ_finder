# Finding Complex Lee-Yang zeros from TrENTo-generated data 

To find the zeros, the complex Bessel function is required. 
I have used the wrapper made by joeydumont in https://github.com/joeydumont/complex_bessel.


## Build and run

To install complex_bessel:
```bash
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/local
make
make install
```
