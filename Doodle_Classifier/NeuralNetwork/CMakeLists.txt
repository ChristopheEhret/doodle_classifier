cmake_minimum_required(VERSION 3.0)

project(lib_RN)

set(SRCS
    ReseauNeuronal.cpp
    Activation.cpp
    VisualisationReseauNeuronal.cpp
)

set(HEADERS 
    ReseauNeuronal.h
    Activation.h
    VisualisationReseauNeuronal.h
)

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

find_package(SFML COMPONENTS graphics window system REQUIRED)

add_library(lib_RN ${SRCS} ${HEADERS})

target_link_libraries(lib_RN sfml-graphics sfml-window sfml-system)
target_link_libraries(lib_RN ${ARMADILLO_LIBRARIES})
