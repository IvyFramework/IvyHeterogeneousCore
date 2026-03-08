PROJECTNAME          = IvyFramework
PACKAGENAME          = IvyHeterogeneousCore

# Parallel compilation: use all available cores.
# -k (keep-going) ensures that a single failing TU does not abort other independent targets.
MAKEFLAGS += -j$(shell nproc) -k


COMPILEPATH          = $(PWD)/

INCLUDEDIR           = $(COMPILEPATH)interface/
BINDIR               = $(COMPILEPATH)bin/
EXEDIR               = $(COMPILEPATH)executables/
TESTEXEDIR           = $(COMPILEPATH)test_executables/
TESTDIR              = $(COMPILEPATH)test/
RUNDIR               = $(COMPILEPATH)


EXTCXXFLAGS   =
EXTLIBS       =

ROOTCFLAGS    =
ROOTLIBS      =

CXX           = g++
CXXINC        = -I$(INCLUDEDIR)
CXXDEFINES    =
CXXVEROPT     = -std=c++20
CXXOPTIM      = -O2

# Check for OpenMP support
OPENMP_TEST := $(shell echo '\#include <omp.h>' | $(CXX) -x c++ -fopenmp -E - > /dev/null 2>&1 && echo "yes")
ifeq ($(OPENMP_TEST),yes)
  OPENMP_FLAGS=-fopenmp
else
  OPENMP_FLAGS=
endif

CXXFLAGS      = -fPIC -g -ftemplate-backtrace-limit=0 $(CXXOPTIM) $(CXXVEROPT) $(OPENMP_FLAGS) $(ROOTCFLAGS) $(CXXDEFINES) $(CXXINC) $(EXTCXXFLAGS)
EXEFLAGS      = $(CXXFLAGS)

# Auto-detect GPU compute capability; fall back to sm_86 if nvidia-smi is unavailable.
GPU_ARCH_RAW := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
ifeq ($(strip $(GPU_ARCH_RAW)),)
  GPU_ARCH_RAW := 86
endif
GPU_GENCODE := -gencode arch=compute_$(GPU_ARCH_RAW),code=[sm_$(GPU_ARCH_RAW),compute_$(GPU_ARCH_RAW)]

ifneq ($(strip $(USE_CUDA)),)
CXX           = nvcc
OPENMP_FLAGS  =
CXXDEFINES    += -D__USE_CUDA__
NVLINKOPTS    = -Xnvlink --suppress-stack-size-warning
CXXFLAGS      = $(GPU_GENCODE) -dc -rdc=true -x cu --cudart=shared $(NVLINKOPTS) -Xcompiler -fPIC -g -ftemplate-backtrace-limit=0 $(CXXOPTIM) $(CXXVEROPT) $(OPENMP_FLAGS) $(ROOTCFLAGS) $(CXXDEFINES) $(CXXINC) $(EXTCXXFLAGS)
EXEFLAGS      = $(filter-out -dc, $(CXXFLAGS))
LIBDLINKFLAGS = $(GPU_GENCODE) $(NVLINKOPTS) -Xcompiler -fPIC
endif

ifeq ($(strip $(USE_CUDA)),)
  # Precompiled header — g++ only. nvcc does not support .gch (nvcc --pch is fatal).
  PCHFILE    = $(INCLUDEDIR)IvyPrecompiledHeader.h
  PCHOUT     = $(PCHFILE).gch
  # EXEFLAGS = $(CXXFLAGS) is a lazy recursive assignment (line 40), so adding
  # -include to CXXFLAGS automatically propagates to EXEFLAGS; no duplicate needed.
  CXXFLAGS  += -include $(PCHFILE)
endif

NLIBS        += $(EXTLIBS)
# Filter out libNew because it leads to floating-point exceptions in versions of ROOT prior to 6.08.02
# See the thread https://root-forum.cern.ch/t/linking-new-library-leads-to-a-floating-point-exception-at-startup/22404
LIBS          = $(filter-out -lNew, $(NLIBS))

BINSCC = $(wildcard $(BINDIR)*.cc)
BINSCXX = $(wildcard $(BINDIR)*.cxx)
TESTSCC = $(wildcard $(TESTDIR)*.cc)
TESTSCXX = $(wildcard $(TESTDIR)*.cxx)
EXESPRIM = $(BINSCC:.cc=) $(BINSCXX:.cxx=)
TESTEXESPRIM = $(TESTSCC:.cc=) $(TESTSCXX:.cxx=)
EXES = $(subst $(BINDIR),$(EXEDIR),$(EXESPRIM))
TESTEXES = $(subst $(TESTDIR),$(TESTEXEDIR),$(TESTEXESPRIM))


.PHONY: all utests help compile clean lib pch distclean
.SILENT: exedirs testexedirs clean $(EXES) $(TESTEXES)


all: $(EXES)


utests: $(TESTEXES)


exedirs:
	mkdir -p $(EXEDIR)


testexedirs:
	mkdir -p $(TESTEXEDIR)


alldirs: exedirs testexedirs


$(EXEDIR)%:: $(BINDIR)%.cc | exedirs
	echo "Compiling $<"; \
	$(CXX) $(EXEFLAGS) -o $@ $< $(LIBS)


$(TESTEXEDIR)%:: $(TESTDIR)%.cc | testexedirs
	echo "Compiling $<"; \
	$(CXX) $(EXEFLAGS) -o $@ $< $(LIBS)


clean:
	rm -rf $(EXEDIR)
	rm -rf $(TESTEXEDIR)
	rm -rf $(LIBDIR)
	rm -f $(INCLUDEDIR)/*.gch
	rm -f $(BINDIR)*.o
	rm -f $(BINDIR)*.so
	rm -f $(BINDIR)*.d
	rm -rf $(RUNDIR)Pdfdata
	rm -f $(RUNDIR)*.DAT
	rm -f $(RUNDIR)*.dat
	rm -f $(RUNDIR)br.sm*
	rm -f $(RUNDIR)*.cc
	rm -f $(RUNDIR)*.o
	rm -f $(RUNDIR)*.so
	rm -f $(RUNDIR)*.d
	rm -f $(RUNDIR)*.pcm
	rm -f $(RUNDIR)*.pyc
	rm -rf $(TESTDIR)Pdfdata
	rm -f $(TESTDIR)*.DAT
	rm -f $(TESTDIR)*.dat
	rm -f $(TESTDIR)br.sm*
	rm -f $(TESTDIR)*.o
	rm -f $(TESTDIR)*.so
	rm -f $(TESTDIR)*.d
	rm -f $(TESTDIR)*.pcm
	rm -f $(TESTDIR)*.pyc


LIBDIR         = $(COMPILEPATH)lib/
LIBNAME        = libIvyHeterogeneousCore
SRCDIR         = $(COMPILEPATH)src/

lib:
	mkdir -p $(LIBDIR)
ifeq ($(strip $(USE_CUDA)),)
	$(CXX) $(CXXFLAGS) -shared -o $(LIBDIR)$(LIBNAME).so $(SRCDIR)IvyHeterogeneousCore.cc
else
	$(CXX) $(CXXFLAGS) -o $(LIBDIR)$(LIBNAME).o $(SRCDIR)IvyHeterogeneousCore.cc
	$(CXX) $(LIBDLINKFLAGS) -dlink -shared $(LIBDIR)$(LIBNAME).o -o $(LIBDIR)$(LIBNAME)_dlink.o
	$(CXX) $(LIBDLINKFLAGS) -shared $(LIBDIR)$(LIBNAME).o $(LIBDIR)$(LIBNAME)_dlink.o -o $(LIBDIR)$(LIBNAME).so -lcudart
	rm -f $(LIBDIR)$(LIBNAME).o $(LIBDIR)$(LIBNAME)_dlink.o
endif


pch:
ifeq ($(strip $(USE_CUDA)),)
	@echo "Building g++ precompiled header (CPU only)..."
	$(CXX) $(filter-out -include $(PCHFILE), $(CXXFLAGS)) -x c++-header -o $(PCHOUT) $(PCHFILE)
	@echo "PCH written: $(PCHOUT)"
else
	@echo "PCH skipped: nvcc does not support .gch precompiled headers."
endif

distclean: clean
	@echo "Removing precompiled header artifacts..."
	rm -f $(INCLUDEDIR)*.gch

include $(DEPS)
