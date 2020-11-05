cmake_minimum_required(VERSION 3.0)

project(Doodle)

#add_subdirectory(NeuralNetwork)

set(SRCS 
    Doodle_Classifier.cpp
)

set(HEADERS 
    stdafx.h
)

find_package(SFML COMPONENTS graphics window system REQUIRED)
#find_package(lib_RN REQUIRED)

add_executable(doodle ${SRCS} ${HEADERS})

target_link_libraries(doodle sfml-graphics sfml-window sfml-system)
target_link_libraries(doodle lib_RN)
