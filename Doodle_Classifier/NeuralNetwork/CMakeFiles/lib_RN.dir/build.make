# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.15

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/christophe/cpp/Projects_Windows/Doodle_Classifier/Doodle_Classifier

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/christophe/cpp/Projects_Windows/Doodle_Classifier/Doodle_Classifier

# Include any dependencies generated for this target.
include NeuralNetwork/CMakeFiles/lib_RN.dir/depend.make

# Include the progress variables for this target.
include NeuralNetwork/CMakeFiles/lib_RN.dir/progress.make

# Include the compile flags for this target's objects.
include NeuralNetwork/CMakeFiles/lib_RN.dir/flags.make

NeuralNetwork/CMakeFiles/lib_RN.dir/ReseauNeuronal.cpp.o: NeuralNetwork/CMakeFiles/lib_RN.dir/flags.make
NeuralNetwork/CMakeFiles/lib_RN.dir/ReseauNeuronal.cpp.o: NeuralNetwork/ReseauNeuronal.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/christophe/cpp/Projects_Windows/Doodle_Classifier/Doodle_Classifier/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object NeuralNetwork/CMakeFiles/lib_RN.dir/ReseauNeuronal.cpp.o"
	cd /home/christophe/cpp/Projects_Windows/Doodle_Classifier/Doodle_Classifier/NeuralNetwork && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/lib_RN.dir/ReseauNeuronal.cpp.o -c /home/christophe/cpp/Projects_Windows/Doodle_Classifier/Doodle_Classifier/NeuralNetwork/ReseauNeuronal.cpp

NeuralNetwork/CMakeFiles/lib_RN.dir/ReseauNeuronal.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/lib_RN.dir/ReseauNeuronal.cpp.i"
	cd /home/christophe/cpp/Projects_Windows/Doodle_Classifier/Doodle_Classifier/NeuralNetwork && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/christophe/cpp/Projects_Windows/Doodle_Classifier/Doodle_Classifier/NeuralNetwork/ReseauNeuronal.cpp > CMakeFiles/lib_RN.dir/ReseauNeuronal.cpp.i

NeuralNetwork/CMakeFiles/lib_RN.dir/ReseauNeuronal.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/lib_RN.dir/ReseauNeuronal.cpp.s"
	cd /home/christophe/cpp/Projects_Windows/Doodle_Classifier/Doodle_Classifier/NeuralNetwork && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/christophe/cpp/Projects_Windows/Doodle_Classifier/Doodle_Classifier/NeuralNetwork/ReseauNeuronal.cpp -o CMakeFiles/lib_RN.dir/ReseauNeuronal.cpp.s

NeuralNetwork/CMakeFiles/lib_RN.dir/Activation.cpp.o: NeuralNetwork/CMakeFiles/lib_RN.dir/flags.make
NeuralNetwork/CMakeFiles/lib_RN.dir/Activation.cpp.o: NeuralNetwork/Activation.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/christophe/cpp/Projects_Windows/Doodle_Classifier/Doodle_Classifier/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object NeuralNetwork/CMakeFiles/lib_RN.dir/Activation.cpp.o"
	cd /home/christophe/cpp/Projects_Windows/Doodle_Classifier/Doodle_Classifier/NeuralNetwork && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/lib_RN.dir/Activation.cpp.o -c /home/christophe/cpp/Projects_Windows/Doodle_Classifier/Doodle_Classifier/NeuralNetwork/Activation.cpp

NeuralNetwork/CMakeFiles/lib_RN.dir/Activation.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/lib_RN.dir/Activation.cpp.i"
	cd /home/christophe/cpp/Projects_Windows/Doodle_Classifier/Doodle_Classifier/NeuralNetwork && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/christophe/cpp/Projects_Windows/Doodle_Classifier/Doodle_Classifier/NeuralNetwork/Activation.cpp > CMakeFiles/lib_RN.dir/Activation.cpp.i

NeuralNetwork/CMakeFiles/lib_RN.dir/Activation.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/lib_RN.dir/Activation.cpp.s"
	cd /home/christophe/cpp/Projects_Windows/Doodle_Classifier/Doodle_Classifier/NeuralNetwork && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/christophe/cpp/Projects_Windows/Doodle_Classifier/Doodle_Classifier/NeuralNetwork/Activation.cpp -o CMakeFiles/lib_RN.dir/Activation.cpp.s

NeuralNetwork/CMakeFiles/lib_RN.dir/VisualisationReseauNeuronal.cpp.o: NeuralNetwork/CMakeFiles/lib_RN.dir/flags.make
NeuralNetwork/CMakeFiles/lib_RN.dir/VisualisationReseauNeuronal.cpp.o: NeuralNetwork/VisualisationReseauNeuronal.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/christophe/cpp/Projects_Windows/Doodle_Classifier/Doodle_Classifier/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object NeuralNetwork/CMakeFiles/lib_RN.dir/VisualisationReseauNeuronal.cpp.o"
	cd /home/christophe/cpp/Projects_Windows/Doodle_Classifier/Doodle_Classifier/NeuralNetwork && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/lib_RN.dir/VisualisationReseauNeuronal.cpp.o -c /home/christophe/cpp/Projects_Windows/Doodle_Classifier/Doodle_Classifier/NeuralNetwork/VisualisationReseauNeuronal.cpp

NeuralNetwork/CMakeFiles/lib_RN.dir/VisualisationReseauNeuronal.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/lib_RN.dir/VisualisationReseauNeuronal.cpp.i"
	cd /home/christophe/cpp/Projects_Windows/Doodle_Classifier/Doodle_Classifier/NeuralNetwork && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/christophe/cpp/Projects_Windows/Doodle_Classifier/Doodle_Classifier/NeuralNetwork/VisualisationReseauNeuronal.cpp > CMakeFiles/lib_RN.dir/VisualisationReseauNeuronal.cpp.i

NeuralNetwork/CMakeFiles/lib_RN.dir/VisualisationReseauNeuronal.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/lib_RN.dir/VisualisationReseauNeuronal.cpp.s"
	cd /home/christophe/cpp/Projects_Windows/Doodle_Classifier/Doodle_Classifier/NeuralNetwork && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/christophe/cpp/Projects_Windows/Doodle_Classifier/Doodle_Classifier/NeuralNetwork/VisualisationReseauNeuronal.cpp -o CMakeFiles/lib_RN.dir/VisualisationReseauNeuronal.cpp.s

# Object files for target lib_RN
lib_RN_OBJECTS = \
"CMakeFiles/lib_RN.dir/ReseauNeuronal.cpp.o" \
"CMakeFiles/lib_RN.dir/Activation.cpp.o" \
"CMakeFiles/lib_RN.dir/VisualisationReseauNeuronal.cpp.o"

# External object files for target lib_RN
lib_RN_EXTERNAL_OBJECTS =

NeuralNetwork/liblib_RN.a: NeuralNetwork/CMakeFiles/lib_RN.dir/ReseauNeuronal.cpp.o
NeuralNetwork/liblib_RN.a: NeuralNetwork/CMakeFiles/lib_RN.dir/Activation.cpp.o
NeuralNetwork/liblib_RN.a: NeuralNetwork/CMakeFiles/lib_RN.dir/VisualisationReseauNeuronal.cpp.o
NeuralNetwork/liblib_RN.a: NeuralNetwork/CMakeFiles/lib_RN.dir/build.make
NeuralNetwork/liblib_RN.a: NeuralNetwork/CMakeFiles/lib_RN.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/christophe/cpp/Projects_Windows/Doodle_Classifier/Doodle_Classifier/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX static library liblib_RN.a"
	cd /home/christophe/cpp/Projects_Windows/Doodle_Classifier/Doodle_Classifier/NeuralNetwork && $(CMAKE_COMMAND) -P CMakeFiles/lib_RN.dir/cmake_clean_target.cmake
	cd /home/christophe/cpp/Projects_Windows/Doodle_Classifier/Doodle_Classifier/NeuralNetwork && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/lib_RN.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
NeuralNetwork/CMakeFiles/lib_RN.dir/build: NeuralNetwork/liblib_RN.a

.PHONY : NeuralNetwork/CMakeFiles/lib_RN.dir/build

NeuralNetwork/CMakeFiles/lib_RN.dir/clean:
	cd /home/christophe/cpp/Projects_Windows/Doodle_Classifier/Doodle_Classifier/NeuralNetwork && $(CMAKE_COMMAND) -P CMakeFiles/lib_RN.dir/cmake_clean.cmake
.PHONY : NeuralNetwork/CMakeFiles/lib_RN.dir/clean

NeuralNetwork/CMakeFiles/lib_RN.dir/depend:
	cd /home/christophe/cpp/Projects_Windows/Doodle_Classifier/Doodle_Classifier && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/christophe/cpp/Projects_Windows/Doodle_Classifier/Doodle_Classifier /home/christophe/cpp/Projects_Windows/Doodle_Classifier/Doodle_Classifier/NeuralNetwork /home/christophe/cpp/Projects_Windows/Doodle_Classifier/Doodle_Classifier /home/christophe/cpp/Projects_Windows/Doodle_Classifier/Doodle_Classifier/NeuralNetwork /home/christophe/cpp/Projects_Windows/Doodle_Classifier/Doodle_Classifier/NeuralNetwork/CMakeFiles/lib_RN.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : NeuralNetwork/CMakeFiles/lib_RN.dir/depend

