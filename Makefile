OS := $(shell uname -s)
ARCH := $(shell uname -m)

CC  = gcc
CXX = g++
CXX_STD = c++20

# Compiler / linker flags
WARNFLAGS  = -Wall -Wextra -Wsuggest-override -Wnon-virtual-dtor -Wno-deprecated-enum-enum-conversion -Wno-volatile
OPTFLAGS   = -O3 -funroll-loops
DEBUGFLAGS = -g
DEFINES    = -DNDEBUG

# Includes (project + dependencies)
PKG_CPPFLAGS = \
  -I./include \
  -I./libs/eigen \
  -I./libs/armadillo/include -DARMA_DONT_USE_WRAPPER -lopenblas -llapack \
  -I./libs/boost \
  -I./libs/fftw/include \
  -I./libs/gsl/include \
  -I./libs/pybind11/include \


# Libraries (local versions)
PKG_LIBS = \
  -L./libs/fftw/lib -lfftw3 -lfftw3_threads \
  -L./libs/gsl/lib -lgsl -lgslcblas \
  -lblas -llapack -pthread

# OpenMP support
ifeq ($(OS),Darwin)
  ifeq ($(ARCH),arm64)
      HOMEBREW_PREFIX = /opt/homebrew
  else
      HOMEBREW_PREFIX = /usr/local
  endif
  PKG_CXXFLAGS = -Xpreprocessor -fopenmp -I$(HOMEBREW_PREFIX)/opt/libomp/include $(OPTFLAGS)
  PKG_LIBS += -L$(HOMEBREW_PREFIX)/opt/libomp/lib -lomp
else ifeq ($(OS),Linux)
  PKG_CXXFLAGS = -fopenmp $(OPTFLAGS)
  PKG_LIBS += -fopenmp
else ifeq ($(OS),Windows)
  PKG_CXXFLAGS = -fopenmp $(OPTFLAGS)
  PKG_LIBS += -fopenmp
endif

# Final flags
CPPFLAGS = $(PKG_CPPFLAGS) $(DEFINES)
CXXFLAGS = -std=$(CXX_STD) $(WARNFLAGS) $(PKG_CXXFLAGS)
LDFLAGS  = $(PKG_LIBS)

# Sources, objects, executables
SRCS    = $(wildcard src/*.cpp)
OBJS    = $(patsubst src/%.cpp,build/%.o,$(SRCS))
HEADERS = $(wildcard include/*.hpp)
Mains   = $(filter src/main%.cpp,$(SRCS))
EXEC  = $(patsubst src/%.cpp,bin/%,$(Mains))


.DEFAULT_GOAL = all

# --- Bootstrap: ensure libs are installed ---
libs/.ready:
	@echo "🔍 Checking dependencies..."
	@if [ ! -d libs ]; then \
	  echo "📦 libs/ not found. Running setup_deps.sh..."; \
	  bash ./setup_deps.sh; \
	fi
	@touch libs/.ready

# --- Build rules ---
all: libs/.ready $(EXEC)

bin/%: build/%.o $(filter-out build/$*.o,$(OBJS))
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

build/%.o: src/%.cpp $(HEADERS) libs/.ready
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

clean:
	$(RM) -rf build bin

distclean: clean
	$(RM) -rf libs
	$(RM) -f make.dep libs/.ready

doc:
	doxygen Doxyfile

# Dependencies
make.dep: $(SRCS)
	$(RM) make.dep
	for f in $(SRCS); do \
		$(CXX) $(CXXFLAGS) $(CPPFLAGS) -MM $$f >> make.dep; \
	done

-include make.dep
