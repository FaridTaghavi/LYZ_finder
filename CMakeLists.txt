cmake_minimum_required(VERSION 3.16)
project(LYZProject LANGUAGES CXX Fortran)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build the AMOS-based complex Bessel wrapper bundled with this project.
set(BUILD_TESTING OFF CACHE BOOL "Build tests" FORCE)
add_subdirectory(complex_bessel)

find_package(ROOT REQUIRED COMPONENTS Core Graf Gpad MathCore RIO Hist MathMore Tree)

add_executable(LYZ main.cpp LYZ.C)

target_link_libraries(LYZ PRIVATE
    complex_bessel::complex_bessel
    ROOT::Core
    ROOT::Graf
    ROOT::MathCore
	ROOT::MathMore
    ROOT::RIO
    ROOT::Hist
    ROOT::Gpad
)
