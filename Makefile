SRCDIR := src
OBJDIR := lib
INCDIR := include


IFLAGS := -I$(INCDIR)/ -isystem /usr/local/include/root -isystem /usr/local/include
CPP_FILES := $(wildcard $(SRCDIR)/*.cpp)
HPP_FILES := $(INCDIR)/*.hpp

#Attach required libraries.
LD_FLAGS := -L/usr/local/lib -lboost_iostreams -lz -lboost_system -lboost_filesystem -flto

ifeq ($(OS),macos)
LD_FLAGS := -framework Accelerate $(LD_FLAGS)
endif

#O3 flag uses string aliasing for code. We want a warning if that happens.
CC_FLAGS :=-std=c++14 -Wall -pedantic -O3 -march=native -Wstrict-aliasing -flto

all: pt doc

debug: CC_FLAGS += -g
debug: LD_FLAGS += -g
debug: pt

pt : lib/main.o lib/pt.o lib/pthelper.o lib/ptdefs.o
	$(CXX) -o $@ $^ $(LD_FLAGS)

lib/%.o : src/%.cpp $(HPP_FILES)
	$(CXX) $(CC_FLAGS) $(IFLAGS) -c -o $@ $<

doc: .Doxyfile $(CPP_FILES) $(HPP_FILES)
	doxygen $^

doc_clean:
	rm -r doc/html/ && rm -r doc/latex/

clean:
	rm -f pt && rm -f pt_test && rm -f $(OBJDIR)/*.o

test: lib/pt.o lib/test.o
	$(CXX) $(LD_FLAGS) -o pt_test lib/test.o lib/pt.o

#for debug reasons.
print-%  : ; @echo $* = $($*)
