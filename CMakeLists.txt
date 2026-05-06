cmake_minimum_required(VERSION 3.16)
project(LYZProject LANGUAGES CXX Fortran)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build the AMOS-based complex Bessel wrapper bundled with this project.
set(BUILD_TESTING OFF CACHE BOOL "Build tests" FORCE)
add_subdirectory(complex_bessel)

find_package(ROOT REQUIRED COMPONENTS Core Graf Gpad MathCore RIO Hist MathMore Tree TreePlayer)
find_package(OpenMP REQUIRED)

root_generate_dictionary(EventDict
    ${CMAKE_CURRENT_SOURCE_DIR}/event_CLASS.h
    LINKDEF ${CMAKE_CURRENT_SOURCE_DIR}/LinkDef.h
)

add_executable(LYZ
    main.cpp
    LYZ.C
    event_CLASS.cpp
    EventDict.cxx
)

target_include_directories(LYZ PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)


target_link_libraries(LYZ PRIVATE
    complex_bessel::complex_bessel
    ROOT::Core
    ROOT::Graf
    ROOT::MathCore
    ROOT::MathMore
    ROOT::RIO
    ROOT::Hist
    ROOT::Gpad
    ROOT::Tree
    ROOT::TreePlayer
    OpenMP::OpenMP_CXX
)
