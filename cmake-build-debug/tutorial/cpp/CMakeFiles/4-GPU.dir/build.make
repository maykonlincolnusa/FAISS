# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.19

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake

# The command to remove a file.
RM = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/liyinbin/github/lambda-search/faiss

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/liyinbin/github/lambda-search/faiss/cmake-build-debug

# Include any dependencies generated for this target.
include tutorial/cpp/CMakeFiles/4-GPU.dir/depend.make

# Include the progress variables for this target.
include tutorial/cpp/CMakeFiles/4-GPU.dir/progress.make

# Include the compile flags for this target's objects.
include tutorial/cpp/CMakeFiles/4-GPU.dir/flags.make

tutorial/cpp/CMakeFiles/4-GPU.dir/4-GPU.cpp.o: tutorial/cpp/CMakeFiles/4-GPU.dir/flags.make
tutorial/cpp/CMakeFiles/4-GPU.dir/4-GPU.cpp.o: ../tutorial/cpp/4-GPU.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/liyinbin/github/lambda-search/faiss/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object tutorial/cpp/CMakeFiles/4-GPU.dir/4-GPU.cpp.o"
	cd /Users/liyinbin/github/lambda-search/faiss/cmake-build-debug/tutorial/cpp && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/4-GPU.dir/4-GPU.cpp.o -c /Users/liyinbin/github/lambda-search/faiss/tutorial/cpp/4-GPU.cpp

tutorial/cpp/CMakeFiles/4-GPU.dir/4-GPU.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/4-GPU.dir/4-GPU.cpp.i"
	cd /Users/liyinbin/github/lambda-search/faiss/cmake-build-debug/tutorial/cpp && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/liyinbin/github/lambda-search/faiss/tutorial/cpp/4-GPU.cpp > CMakeFiles/4-GPU.dir/4-GPU.cpp.i

tutorial/cpp/CMakeFiles/4-GPU.dir/4-GPU.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/4-GPU.dir/4-GPU.cpp.s"
	cd /Users/liyinbin/github/lambda-search/faiss/cmake-build-debug/tutorial/cpp && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/liyinbin/github/lambda-search/faiss/tutorial/cpp/4-GPU.cpp -o CMakeFiles/4-GPU.dir/4-GPU.cpp.s

# Object files for target 4-GPU
4__GPU_OBJECTS = \
"CMakeFiles/4-GPU.dir/4-GPU.cpp.o"

# External object files for target 4-GPU
4__GPU_EXTERNAL_OBJECTS =

tutorial/cpp/4-GPU: tutorial/cpp/CMakeFiles/4-GPU.dir/4-GPU.cpp.o
tutorial/cpp/4-GPU: tutorial/cpp/CMakeFiles/4-GPU.dir/build.make
tutorial/cpp/4-GPU: faiss/libfaiss.a
tutorial/cpp/4-GPU: /usr/local/lib/libomp.dylib
tutorial/cpp/4-GPU: tutorial/cpp/CMakeFiles/4-GPU.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/liyinbin/github/lambda-search/faiss/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable 4-GPU"
	cd /Users/liyinbin/github/lambda-search/faiss/cmake-build-debug/tutorial/cpp && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/4-GPU.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tutorial/cpp/CMakeFiles/4-GPU.dir/build: tutorial/cpp/4-GPU

.PHONY : tutorial/cpp/CMakeFiles/4-GPU.dir/build

tutorial/cpp/CMakeFiles/4-GPU.dir/clean:
	cd /Users/liyinbin/github/lambda-search/faiss/cmake-build-debug/tutorial/cpp && $(CMAKE_COMMAND) -P CMakeFiles/4-GPU.dir/cmake_clean.cmake
.PHONY : tutorial/cpp/CMakeFiles/4-GPU.dir/clean

tutorial/cpp/CMakeFiles/4-GPU.dir/depend:
	cd /Users/liyinbin/github/lambda-search/faiss/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/liyinbin/github/lambda-search/faiss /Users/liyinbin/github/lambda-search/faiss/tutorial/cpp /Users/liyinbin/github/lambda-search/faiss/cmake-build-debug /Users/liyinbin/github/lambda-search/faiss/cmake-build-debug/tutorial/cpp /Users/liyinbin/github/lambda-search/faiss/cmake-build-debug/tutorial/cpp/CMakeFiles/4-GPU.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tutorial/cpp/CMakeFiles/4-GPU.dir/depend

