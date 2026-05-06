# LYZ project build notes

The complex function is needed in this project. 
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

To install LYZ finder, in the same build folder:
```bash
cmake ..
make
```
