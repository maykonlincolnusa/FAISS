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
CMAKE_COMMAND = /Users/liyinbin/miniconda3/envs/flare-dev/bin/cmake

# The command to remove a file.
RM = /Users/liyinbin/miniconda3/envs/flare-dev/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/liyinbin/github/lambda-search/faiss

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/liyinbin/github/lambda-search/faiss/build

# Include any dependencies generated for this target.
include demos/CMakeFiles/demo_sift1M.dir/depend.make

# Include the progress variables for this target.
include demos/CMakeFiles/demo_sift1M.dir/progress.make

# Include the compile flags for this target's objects.
include demos/CMakeFiles/demo_sift1M.dir/flags.make

demos/CMakeFiles/demo_sift1M.dir/demo_sift1M.cpp.o: demos/CMakeFiles/demo_sift1M.dir/flags.make
demos/CMakeFiles/demo_sift1M.dir/demo_sift1M.cpp.o: ../demos/demo_sift1M.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/liyinbin/github/lambda-search/faiss/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object demos/CMakeFiles/demo_sift1M.dir/demo_sift1M.cpp.o"
	cd /Users/liyinbin/github/lambda-search/faiss/build/demos && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/demo_sift1M.dir/demo_sift1M.cpp.o -c /Users/liyinbin/github/lambda-search/faiss/demos/demo_sift1M.cpp

demos/CMakeFiles/demo_sift1M.dir/demo_sift1M.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/demo_sift1M.dir/demo_sift1M.cpp.i"
	cd /Users/liyinbin/github/lambda-search/faiss/build/demos && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/liyinbin/github/lambda-search/faiss/demos/demo_sift1M.cpp > CMakeFiles/demo_sift1M.dir/demo_sift1M.cpp.i

demos/CMakeFiles/demo_sift1M.dir/demo_sift1M.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/demo_sift1M.dir/demo_sift1M.cpp.s"
	cd /Users/liyinbin/github/lambda-search/faiss/build/demos && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/liyinbin/github/lambda-search/faiss/demos/demo_sift1M.cpp -o CMakeFiles/demo_sift1M.dir/demo_sift1M.cpp.s

# Object files for target demo_sift1M
demo_sift1M_OBJECTS = \
"CMakeFiles/demo_sift1M.dir/demo_sift1M.cpp.o"

# External object files for target demo_sift1M
demo_sift1M_EXTERNAL_OBJECTS =

demos/demo_sift1M: demos/CMakeFiles/demo_sift1M.dir/demo_sift1M.cpp.o
demos/demo_sift1M: demos/CMakeFiles/demo_sift1M.dir/build.make
demos/demo_sift1M: faiss/libfaiss.a
demos/demo_sift1M: /usr/local/lib/libomp.dylib
demos/demo_sift1M: /Users/liyinbin/miniconda3/envs/flare-dev/lib/libmkl_intel_lp64.dylib
demos/demo_sift1M: /Users/liyinbin/miniconda3/envs/flare-dev/lib/libmkl_intel_thread.dylib
demos/demo_sift1M: /Users/liyinbin/miniconda3/envs/flare-dev/lib/libmkl_core.dylib
demos/demo_sift1M: /Users/liyinbin/miniconda3/envs/flare-dev/lib/libiomp5.dylib
demos/demo_sift1M: demos/CMakeFiles/demo_sift1M.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/liyinbin/github/lambda-search/faiss/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable demo_sift1M"
	cd /Users/liyinbin/github/lambda-search/faiss/build/demos && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/demo_sift1M.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
demos/CMakeFiles/demo_sift1M.dir/build: demos/demo_sift1M

.PHONY : demos/CMakeFiles/demo_sift1M.dir/build

demos/CMakeFiles/demo_sift1M.dir/clean:
	cd /Users/liyinbin/github/lambda-search/faiss/build/demos && $(CMAKE_COMMAND) -P CMakeFiles/demo_sift1M.dir/cmake_clean.cmake
.PHONY : demos/CMakeFiles/demo_sift1M.dir/clean

demos/CMakeFiles/demo_sift1M.dir/depend:
	cd /Users/liyinbin/github/lambda-search/faiss/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/liyinbin/github/lambda-search/faiss /Users/liyinbin/github/lambda-search/faiss/demos /Users/liyinbin/github/lambda-search/faiss/build /Users/liyinbin/github/lambda-search/faiss/build/demos /Users/liyinbin/github/lambda-search/faiss/build/demos/CMakeFiles/demo_sift1M.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : demos/CMakeFiles/demo_sift1M.dir/depend

