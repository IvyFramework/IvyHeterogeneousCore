PROJECTNAME          = IvyFramework
PACKAGENAME          = IvyHeterogeneousCore


COMPILEPATH          = $(PWD)/

INCLUDEDIR           = $(COMPILEPATH)interface/
BINDIR               = $(COMPILEPATH)bin/
EXEDIR               = $(COMPILEPATH)executables/
TESTDIR              = $(COMPILEPATH)test/
RUNDIR               = $(COMPILEPATH)


EXTCXXFLAGS   =
EXTLIBS       =

ROOTCFLAGS    =
ROOTLIBS      =

CXX           = g++
CXXINC        = -I$(INCLUDEDIR)
CXXDEFINES    =
CXXFLAGS      = -fPIC -g -O2 -std=c++20 $(ROOTCFLAGS) $(CXXDEFINES) $(CXXINC) $(EXTCXXFLAGS)
EXEFLAGS      = $(CXXFLAGS)

ifneq ($(strip $(USE_CUDA)),)
CXX           = nvcc
CXXDEFINES    += -D__USE_CUDA__
CXXFLAGS      = -dc -rdc=true -x cu --cudart=shared -Xcompiler -fPIC -g -O2 -std=c++20 $(ROOTCFLAGS) $(CXXDEFINES) $(CXXINC) $(EXTCXXFLAGS)
EXEFLAGS      = $(filter-out -dc, $(CXXFLAGS))
endif

NLIBS        += $(EXTLIBS)
# Filter out libNew because it leads to floating-point exceptions in versions of ROOT prior to 6.08.02
# See the thread https://root-forum.cern.ch/t/linking-new-library-leads-to-a-floating-point-exception-at-startup/22404
LIBS          = $(filter-out -lNew, $(NLIBS))

BINSCC = $(wildcard $(BINDIR)*.cc)
BINSCXX = $(wildcard $(BINDIR)*.cxx)
EXESPRIM = $(BINSCC:.cc=) $(BINSCXX:.cxx=)
EXES = $(subst $(BINDIR),$(EXEDIR),$(EXESPRIM))


.PHONY: all help compile clean
.SILENT: alldirs clean $(EXES)


all: $(EXES)


alldirs:
	mkdir -p $(EXEDIR)


$(EXEDIR)%::	$(BINDIR)%.cc | alldirs
	echo "Compiling $<"; \
	$(CXX) $(EXEFLAGS) -o $@ $< $(LIBS)


clean:
	rm -rf $(EXEDIR)
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


include $(DEPS)
