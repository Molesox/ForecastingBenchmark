# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Default target executed when no arguments are given to make.
default_target: all

.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:


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
CMAKE_SOURCE_DIR = /mnt/c/Users/Daniel/Desktop/FBthesis/Algorithms/Algorithms

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /mnt/c/Users/Daniel/Desktop/FBthesis/Algorithms/Algorithms

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/bin/cmake -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache

.PHONY : rebuild_cache/fast

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "No interactive CMake dialog available..."
	/usr/bin/cmake -E echo No\ interactive\ CMake\ dialog\ available.
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache

.PHONY : edit_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /mnt/c/Users/Daniel/Desktop/FBthesis/Algorithms/Algorithms/CMakeFiles /mnt/c/Users/Daniel/Desktop/FBthesis/Algorithms/Algorithms/CMakeFiles/progress.marks
	$(MAKE) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /mnt/c/Users/Daniel/Desktop/FBthesis/Algorithms/Algorithms/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean

.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named Algorithms

# Build rule for target.
Algorithms: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 Algorithms
.PHONY : Algorithms

# fast build rule for target.
Algorithms/fast:
	$(MAKE) -f CMakeFiles/Algorithms.dir/build.make CMakeFiles/Algorithms.dir/build
.PHONY : Algorithms/fast

src/LSRN/LatentSpaceRN.o: src/LSRN/LatentSpaceRN.cpp.o

.PHONY : src/LSRN/LatentSpaceRN.o

# target to build an object file
src/LSRN/LatentSpaceRN.cpp.o:
	$(MAKE) -f CMakeFiles/Algorithms.dir/build.make CMakeFiles/Algorithms.dir/src/LSRN/LatentSpaceRN.cpp.o
.PHONY : src/LSRN/LatentSpaceRN.cpp.o

src/LSRN/LatentSpaceRN.i: src/LSRN/LatentSpaceRN.cpp.i

.PHONY : src/LSRN/LatentSpaceRN.i

# target to preprocess a source file
src/LSRN/LatentSpaceRN.cpp.i:
	$(MAKE) -f CMakeFiles/Algorithms.dir/build.make CMakeFiles/Algorithms.dir/src/LSRN/LatentSpaceRN.cpp.i
.PHONY : src/LSRN/LatentSpaceRN.cpp.i

src/LSRN/LatentSpaceRN.s: src/LSRN/LatentSpaceRN.cpp.s

.PHONY : src/LSRN/LatentSpaceRN.s

# target to generate assembly for a file
src/LSRN/LatentSpaceRN.cpp.s:
	$(MAKE) -f CMakeFiles/Algorithms.dir/build.make CMakeFiles/Algorithms.dir/src/LSRN/LatentSpaceRN.cpp.s
.PHONY : src/LSRN/LatentSpaceRN.cpp.s

src/MAME/MAME_svd.o: src/MAME/MAME_svd.cpp.o

.PHONY : src/MAME/MAME_svd.o

# target to build an object file
src/MAME/MAME_svd.cpp.o:
	$(MAKE) -f CMakeFiles/Algorithms.dir/build.make CMakeFiles/Algorithms.dir/src/MAME/MAME_svd.cpp.o
.PHONY : src/MAME/MAME_svd.cpp.o

src/MAME/MAME_svd.i: src/MAME/MAME_svd.cpp.i

.PHONY : src/MAME/MAME_svd.i

# target to preprocess a source file
src/MAME/MAME_svd.cpp.i:
	$(MAKE) -f CMakeFiles/Algorithms.dir/build.make CMakeFiles/Algorithms.dir/src/MAME/MAME_svd.cpp.i
.PHONY : src/MAME/MAME_svd.cpp.i

src/MAME/MAME_svd.s: src/MAME/MAME_svd.cpp.s

.PHONY : src/MAME/MAME_svd.s

# target to generate assembly for a file
src/MAME/MAME_svd.cpp.s:
	$(MAKE) -f CMakeFiles/Algorithms.dir/build.make CMakeFiles/Algorithms.dir/src/MAME/MAME_svd.cpp.s
.PHONY : src/MAME/MAME_svd.cpp.s

src/OATS/OATS_ogd.o: src/OATS/OATS_ogd.cpp.o

.PHONY : src/OATS/OATS_ogd.o

# target to build an object file
src/OATS/OATS_ogd.cpp.o:
	$(MAKE) -f CMakeFiles/Algorithms.dir/build.make CMakeFiles/Algorithms.dir/src/OATS/OATS_ogd.cpp.o
.PHONY : src/OATS/OATS_ogd.cpp.o

src/OATS/OATS_ogd.i: src/OATS/OATS_ogd.cpp.i

.PHONY : src/OATS/OATS_ogd.i

# target to preprocess a source file
src/OATS/OATS_ogd.cpp.i:
	$(MAKE) -f CMakeFiles/Algorithms.dir/build.make CMakeFiles/Algorithms.dir/src/OATS/OATS_ogd.cpp.i
.PHONY : src/OATS/OATS_ogd.cpp.i

src/OATS/OATS_ogd.s: src/OATS/OATS_ogd.cpp.s

.PHONY : src/OATS/OATS_ogd.s

# target to generate assembly for a file
src/OATS/OATS_ogd.cpp.s:
	$(MAKE) -f CMakeFiles/Algorithms.dir/build.make CMakeFiles/Algorithms.dir/src/OATS/OATS_ogd.cpp.s
.PHONY : src/OATS/OATS_ogd.cpp.s

src/OATS/TemplateOATS.o: src/OATS/TemplateOATS.cpp.o

.PHONY : src/OATS/TemplateOATS.o

# target to build an object file
src/OATS/TemplateOATS.cpp.o:
	$(MAKE) -f CMakeFiles/Algorithms.dir/build.make CMakeFiles/Algorithms.dir/src/OATS/TemplateOATS.cpp.o
.PHONY : src/OATS/TemplateOATS.cpp.o

src/OATS/TemplateOATS.i: src/OATS/TemplateOATS.cpp.i

.PHONY : src/OATS/TemplateOATS.i

# target to preprocess a source file
src/OATS/TemplateOATS.cpp.i:
	$(MAKE) -f CMakeFiles/Algorithms.dir/build.make CMakeFiles/Algorithms.dir/src/OATS/TemplateOATS.cpp.i
.PHONY : src/OATS/TemplateOATS.cpp.i

src/OATS/TemplateOATS.s: src/OATS/TemplateOATS.cpp.s

.PHONY : src/OATS/TemplateOATS.s

# target to generate assembly for a file
src/OATS/TemplateOATS.cpp.s:
	$(MAKE) -f CMakeFiles/Algorithms.dir/build.make CMakeFiles/Algorithms.dir/src/OATS/TemplateOATS.cpp.s
.PHONY : src/OATS/TemplateOATS.cpp.s

src/OMF/FixedPenalty.o: src/OMF/FixedPenalty.cpp.o

.PHONY : src/OMF/FixedPenalty.o

# target to build an object file
src/OMF/FixedPenalty.cpp.o:
	$(MAKE) -f CMakeFiles/Algorithms.dir/build.make CMakeFiles/Algorithms.dir/src/OMF/FixedPenalty.cpp.o
.PHONY : src/OMF/FixedPenalty.cpp.o

src/OMF/FixedPenalty.i: src/OMF/FixedPenalty.cpp.i

.PHONY : src/OMF/FixedPenalty.i

# target to preprocess a source file
src/OMF/FixedPenalty.cpp.i:
	$(MAKE) -f CMakeFiles/Algorithms.dir/build.make CMakeFiles/Algorithms.dir/src/OMF/FixedPenalty.cpp.i
.PHONY : src/OMF/FixedPenalty.cpp.i

src/OMF/FixedPenalty.s: src/OMF/FixedPenalty.cpp.s

.PHONY : src/OMF/FixedPenalty.s

# target to generate assembly for a file
src/OMF/FixedPenalty.cpp.s:
	$(MAKE) -f CMakeFiles/Algorithms.dir/build.make CMakeFiles/Algorithms.dir/src/OMF/FixedPenalty.cpp.s
.PHONY : src/OMF/FixedPenalty.cpp.s

src/OMF/FixedTolerance.o: src/OMF/FixedTolerance.cpp.o

.PHONY : src/OMF/FixedTolerance.o

# target to build an object file
src/OMF/FixedTolerance.cpp.o:
	$(MAKE) -f CMakeFiles/Algorithms.dir/build.make CMakeFiles/Algorithms.dir/src/OMF/FixedTolerance.cpp.o
.PHONY : src/OMF/FixedTolerance.cpp.o

src/OMF/FixedTolerance.i: src/OMF/FixedTolerance.cpp.i

.PHONY : src/OMF/FixedTolerance.i

# target to preprocess a source file
src/OMF/FixedTolerance.cpp.i:
	$(MAKE) -f CMakeFiles/Algorithms.dir/build.make CMakeFiles/Algorithms.dir/src/OMF/FixedTolerance.cpp.i
.PHONY : src/OMF/FixedTolerance.cpp.i

src/OMF/FixedTolerance.s: src/OMF/FixedTolerance.cpp.s

.PHONY : src/OMF/FixedTolerance.s

# target to generate assembly for a file
src/OMF/FixedTolerance.cpp.s:
	$(MAKE) -f CMakeFiles/Algorithms.dir/build.make CMakeFiles/Algorithms.dir/src/OMF/FixedTolerance.cpp.s
.PHONY : src/OMF/FixedTolerance.cpp.s

src/OMF/TemplateOMF.o: src/OMF/TemplateOMF.cpp.o

.PHONY : src/OMF/TemplateOMF.o

# target to build an object file
src/OMF/TemplateOMF.cpp.o:
	$(MAKE) -f CMakeFiles/Algorithms.dir/build.make CMakeFiles/Algorithms.dir/src/OMF/TemplateOMF.cpp.o
.PHONY : src/OMF/TemplateOMF.cpp.o

src/OMF/TemplateOMF.i: src/OMF/TemplateOMF.cpp.i

.PHONY : src/OMF/TemplateOMF.i

# target to preprocess a source file
src/OMF/TemplateOMF.cpp.i:
	$(MAKE) -f CMakeFiles/Algorithms.dir/build.make CMakeFiles/Algorithms.dir/src/OMF/TemplateOMF.cpp.i
.PHONY : src/OMF/TemplateOMF.cpp.i

src/OMF/TemplateOMF.s: src/OMF/TemplateOMF.cpp.s

.PHONY : src/OMF/TemplateOMF.s

# target to generate assembly for a file
src/OMF/TemplateOMF.cpp.s:
	$(MAKE) -f CMakeFiles/Algorithms.dir/build.make CMakeFiles/Algorithms.dir/src/OMF/TemplateOMF.cpp.s
.PHONY : src/OMF/TemplateOMF.cpp.s

src/OMF/ZeroTolerance.o: src/OMF/ZeroTolerance.cpp.o

.PHONY : src/OMF/ZeroTolerance.o

# target to build an object file
src/OMF/ZeroTolerance.cpp.o:
	$(MAKE) -f CMakeFiles/Algorithms.dir/build.make CMakeFiles/Algorithms.dir/src/OMF/ZeroTolerance.cpp.o
.PHONY : src/OMF/ZeroTolerance.cpp.o

src/OMF/ZeroTolerance.i: src/OMF/ZeroTolerance.cpp.i

.PHONY : src/OMF/ZeroTolerance.i

# target to preprocess a source file
src/OMF/ZeroTolerance.cpp.i:
	$(MAKE) -f CMakeFiles/Algorithms.dir/build.make CMakeFiles/Algorithms.dir/src/OMF/ZeroTolerance.cpp.i
.PHONY : src/OMF/ZeroTolerance.cpp.i

src/OMF/ZeroTolerance.s: src/OMF/ZeroTolerance.cpp.s

.PHONY : src/OMF/ZeroTolerance.s

# target to generate assembly for a file
src/OMF/ZeroTolerance.cpp.s:
	$(MAKE) -f CMakeFiles/Algorithms.dir/build.make CMakeFiles/Algorithms.dir/src/OMF/ZeroTolerance.cpp.s
.PHONY : src/OMF/ZeroTolerance.cpp.s

src/OTS/OTS_gsr.o: src/OTS/OTS_gsr.cpp.o

.PHONY : src/OTS/OTS_gsr.o

# target to build an object file
src/OTS/OTS_gsr.cpp.o:
	$(MAKE) -f CMakeFiles/Algorithms.dir/build.make CMakeFiles/Algorithms.dir/src/OTS/OTS_gsr.cpp.o
.PHONY : src/OTS/OTS_gsr.cpp.o

src/OTS/OTS_gsr.i: src/OTS/OTS_gsr.cpp.i

.PHONY : src/OTS/OTS_gsr.i

# target to preprocess a source file
src/OTS/OTS_gsr.cpp.i:
	$(MAKE) -f CMakeFiles/Algorithms.dir/build.make CMakeFiles/Algorithms.dir/src/OTS/OTS_gsr.cpp.i
.PHONY : src/OTS/OTS_gsr.cpp.i

src/OTS/OTS_gsr.s: src/OTS/OTS_gsr.cpp.s

.PHONY : src/OTS/OTS_gsr.s

# target to generate assembly for a file
src/OTS/OTS_gsr.cpp.s:
	$(MAKE) -f CMakeFiles/Algorithms.dir/build.make CMakeFiles/Algorithms.dir/src/OTS/OTS_gsr.cpp.s
.PHONY : src/OTS/OTS_gsr.cpp.s

src/OTS/OTS_ogd.o: src/OTS/OTS_ogd.cpp.o

.PHONY : src/OTS/OTS_ogd.o

# target to build an object file
src/OTS/OTS_ogd.cpp.o:
	$(MAKE) -f CMakeFiles/Algorithms.dir/build.make CMakeFiles/Algorithms.dir/src/OTS/OTS_ogd.cpp.o
.PHONY : src/OTS/OTS_ogd.cpp.o

src/OTS/OTS_ogd.i: src/OTS/OTS_ogd.cpp.i

.PHONY : src/OTS/OTS_ogd.i

# target to preprocess a source file
src/OTS/OTS_ogd.cpp.i:
	$(MAKE) -f CMakeFiles/Algorithms.dir/build.make CMakeFiles/Algorithms.dir/src/OTS/OTS_ogd.cpp.i
.PHONY : src/OTS/OTS_ogd.cpp.i

src/OTS/OTS_ogd.s: src/OTS/OTS_ogd.cpp.s

.PHONY : src/OTS/OTS_ogd.s

# target to generate assembly for a file
src/OTS/OTS_ogd.cpp.s:
	$(MAKE) -f CMakeFiles/Algorithms.dir/build.make CMakeFiles/Algorithms.dir/src/OTS/OTS_ogd.cpp.s
.PHONY : src/OTS/OTS_ogd.cpp.s

src/OTS/TemplateOTS.o: src/OTS/TemplateOTS.cpp.o

.PHONY : src/OTS/TemplateOTS.o

# target to build an object file
src/OTS/TemplateOTS.cpp.o:
	$(MAKE) -f CMakeFiles/Algorithms.dir/build.make CMakeFiles/Algorithms.dir/src/OTS/TemplateOTS.cpp.o
.PHONY : src/OTS/TemplateOTS.cpp.o

src/OTS/TemplateOTS.i: src/OTS/TemplateOTS.cpp.i

.PHONY : src/OTS/TemplateOTS.i

# target to preprocess a source file
src/OTS/TemplateOTS.cpp.i:
	$(MAKE) -f CMakeFiles/Algorithms.dir/build.make CMakeFiles/Algorithms.dir/src/OTS/TemplateOTS.cpp.i
.PHONY : src/OTS/TemplateOTS.cpp.i

src/OTS/TemplateOTS.s: src/OTS/TemplateOTS.cpp.s

.PHONY : src/OTS/TemplateOTS.s

# target to generate assembly for a file
src/OTS/TemplateOTS.cpp.s:
	$(MAKE) -f CMakeFiles/Algorithms.dir/build.make CMakeFiles/Algorithms.dir/src/OTS/TemplateOTS.cpp.s
.PHONY : src/OTS/TemplateOTS.cpp.s

src/TRMF/TRMF.o: src/TRMF/TRMF.cpp.o

.PHONY : src/TRMF/TRMF.o

# target to build an object file
src/TRMF/TRMF.cpp.o:
	$(MAKE) -f CMakeFiles/Algorithms.dir/build.make CMakeFiles/Algorithms.dir/src/TRMF/TRMF.cpp.o
.PHONY : src/TRMF/TRMF.cpp.o

src/TRMF/TRMF.i: src/TRMF/TRMF.cpp.i

.PHONY : src/TRMF/TRMF.i

# target to preprocess a source file
src/TRMF/TRMF.cpp.i:
	$(MAKE) -f CMakeFiles/Algorithms.dir/build.make CMakeFiles/Algorithms.dir/src/TRMF/TRMF.cpp.i
.PHONY : src/TRMF/TRMF.cpp.i

src/TRMF/TRMF.s: src/TRMF/TRMF.cpp.s

.PHONY : src/TRMF/TRMF.s

# target to generate assembly for a file
src/TRMF/TRMF.cpp.s:
	$(MAKE) -f CMakeFiles/Algorithms.dir/build.make CMakeFiles/Algorithms.dir/src/TRMF/TRMF.cpp.s
.PHONY : src/TRMF/TRMF.cpp.s

src/main.o: src/main.cpp.o

.PHONY : src/main.o

# target to build an object file
src/main.cpp.o:
	$(MAKE) -f CMakeFiles/Algorithms.dir/build.make CMakeFiles/Algorithms.dir/src/main.cpp.o
.PHONY : src/main.cpp.o

src/main.i: src/main.cpp.i

.PHONY : src/main.i

# target to preprocess a source file
src/main.cpp.i:
	$(MAKE) -f CMakeFiles/Algorithms.dir/build.make CMakeFiles/Algorithms.dir/src/main.cpp.i
.PHONY : src/main.cpp.i

src/main.s: src/main.cpp.s

.PHONY : src/main.s

# target to generate assembly for a file
src/main.cpp.s:
	$(MAKE) -f CMakeFiles/Algorithms.dir/build.make CMakeFiles/Algorithms.dir/src/main.cpp.s
.PHONY : src/main.cpp.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... rebuild_cache"
	@echo "... Algorithms"
	@echo "... edit_cache"
	@echo "... src/LSRN/LatentSpaceRN.o"
	@echo "... src/LSRN/LatentSpaceRN.i"
	@echo "... src/LSRN/LatentSpaceRN.s"
	@echo "... src/MAME/MAME_svd.o"
	@echo "... src/MAME/MAME_svd.i"
	@echo "... src/MAME/MAME_svd.s"
	@echo "... src/OATS/OATS_ogd.o"
	@echo "... src/OATS/OATS_ogd.i"
	@echo "... src/OATS/OATS_ogd.s"
	@echo "... src/OATS/TemplateOATS.o"
	@echo "... src/OATS/TemplateOATS.i"
	@echo "... src/OATS/TemplateOATS.s"
	@echo "... src/OMF/FixedPenalty.o"
	@echo "... src/OMF/FixedPenalty.i"
	@echo "... src/OMF/FixedPenalty.s"
	@echo "... src/OMF/FixedTolerance.o"
	@echo "... src/OMF/FixedTolerance.i"
	@echo "... src/OMF/FixedTolerance.s"
	@echo "... src/OMF/TemplateOMF.o"
	@echo "... src/OMF/TemplateOMF.i"
	@echo "... src/OMF/TemplateOMF.s"
	@echo "... src/OMF/ZeroTolerance.o"
	@echo "... src/OMF/ZeroTolerance.i"
	@echo "... src/OMF/ZeroTolerance.s"
	@echo "... src/OTS/OTS_gsr.o"
	@echo "... src/OTS/OTS_gsr.i"
	@echo "... src/OTS/OTS_gsr.s"
	@echo "... src/OTS/OTS_ogd.o"
	@echo "... src/OTS/OTS_ogd.i"
	@echo "... src/OTS/OTS_ogd.s"
	@echo "... src/OTS/TemplateOTS.o"
	@echo "... src/OTS/TemplateOTS.i"
	@echo "... src/OTS/TemplateOTS.s"
	@echo "... src/TRMF/TRMF.o"
	@echo "... src/TRMF/TRMF.i"
	@echo "... src/TRMF/TRMF.s"
	@echo "... src/main.o"
	@echo "... src/main.i"
	@echo "... src/main.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

