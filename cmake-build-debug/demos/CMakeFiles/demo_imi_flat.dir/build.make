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
include demos/CMakeFiles/demo_imi_flat.dir/depend.make

# Include the progress variables for this target.
include demos/CMakeFiles/demo_imi_flat.dir/progress.make

# Include the compile flags for this target's objects.
include demos/CMakeFiles/demo_imi_flat.dir/flags.make

demos/CMakeFiles/demo_imi_flat.dir/demo_imi_flat.cpp.o: demos/CMakeFiles/demo_imi_flat.dir/flags.make
demos/CMakeFiles/demo_imi_flat.dir/demo_imi_flat.cpp.o: ../demos/demo_imi_flat.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/liyinbin/github/lambda-search/faiss/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object demos/CMakeFiles/demo_imi_flat.dir/demo_imi_flat.cpp.o"
	cd /Users/liyinbin/github/lambda-search/faiss/cmake-build-debug/demos && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/demo_imi_flat.dir/demo_imi_flat.cpp.o -c /Users/liyinbin/github/lambda-search/faiss/demos/demo_imi_flat.cpp

demos/CMakeFiles/demo_imi_flat.dir/demo_imi_flat.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/demo_imi_flat.dir/demo_imi_flat.cpp.i"
	cd /Users/liyinbin/github/lambda-search/faiss/cmake-build-debug/demos && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/liyinbin/github/lambda-search/faiss/demos/demo_imi_flat.cpp > CMakeFiles/demo_imi_flat.dir/demo_imi_flat.cpp.i

demos/CMakeFiles/demo_imi_flat.dir/demo_imi_flat.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/demo_imi_flat.dir/demo_imi_flat.cpp.s"
	cd /Users/liyinbin/github/lambda-search/faiss/cmake-build-debug/demos && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/liyinbin/github/lambda-search/faiss/demos/demo_imi_flat.cpp -o CMakeFiles/demo_imi_flat.dir/demo_imi_flat.cpp.s

# Object files for target demo_imi_flat
demo_imi_flat_OBJECTS = \
"CMakeFiles/demo_imi_flat.dir/demo_imi_flat.cpp.o"

# External object files for target demo_imi_flat
demo_imi_flat_EXTERNAL_OBJECTS =

demos/demo_imi_flat: demos/CMakeFiles/demo_imi_flat.dir/demo_imi_flat.cpp.o
demos/demo_imi_flat: demos/CMakeFiles/demo_imi_flat.dir/build.make
demos/demo_imi_flat: faiss/libfaiss.a
demos/demo_imi_flat: /usr/local/lib/libomp.dylib
demos/demo_imi_flat: demos/CMakeFiles/demo_imi_flat.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/liyinbin/github/lambda-search/faiss/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable demo_imi_flat"
	cd /Users/liyinbin/github/lambda-search/faiss/cmake-build-debug/demos && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/demo_imi_flat.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
demos/CMakeFiles/demo_imi_flat.dir/build: demos/demo_imi_flat

.PHONY : demos/CMakeFiles/demo_imi_flat.dir/build

demos/CMakeFiles/demo_imi_flat.dir/clean:
	cd /Users/liyinbin/github/lambda-search/faiss/cmake-build-debug/demos && $(CMAKE_COMMAND) -P CMakeFiles/demo_imi_flat.dir/cmake_clean.cmake
.PHONY : demos/CMakeFiles/demo_imi_flat.dir/clean

demos/CMakeFiles/demo_imi_flat.dir/depend:
	cd /Users/liyinbin/github/lambda-search/faiss/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/liyinbin/github/lambda-search/faiss /Users/liyinbin/github/lambda-search/faiss/demos /Users/liyinbin/github/lambda-search/faiss/cmake-build-debug /Users/liyinbin/github/lambda-search/faiss/cmake-build-debug/demos /Users/liyinbin/github/lambda-search/faiss/cmake-build-debug/demos/CMakeFiles/demo_imi_flat.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : demos/CMakeFiles/demo_imi_flat.dir/depend

