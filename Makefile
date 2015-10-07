CC := clang++
SRCDIR := src
OBJDIR := lib
INCDIR := include
IFLAGS := -I$(INCDIR)/ -I/usr/local/include/root -isystem /usr/local/include

CPP_FILES := $(wildcard $(SRCDIR)/*.cpp)
HPP_FILES := $(INCDIR)/*.hpp

#Attach required libraries for root.
ROOT_LIBS := $(shell root-config --glibs)
LD_FLAGS := $(ROOT_LIBS) -L/usr/local/lib -lboost_iostreams -lz -flto

ifeq (macos,macos)
LD_FLAGS := -framework Accelerate $(LD_FLAGS)
endif

#O3 flag uses string aliasing for code. We want a warning if that happens.
CC_FLAGS :=-std=c++14 -Wall -pedantic -O3 -march=native -Wstrict-aliasing -flto

all: pt doc

debug: CC_FLAGS += -g
debug: LD_FLAGS += -g
debug: pt

pt : lib/main.o lib/pt.o lib/pthelper.o lib/ptdefs.o
	$(CC) -o $@ $^ $(LD_FLAGS)

lib/%.o : src/%.cpp $(HPP_FILES)
	$(CC) $(CC_FLAGS) $(IFLAGS) -c -o $@ $<

doc: .Doxyfile $(CPP_FILES) $(HPP_FILES)
	doxygen $^

doc_clean:
	rm -r doc/html/ && rm -r doc/latex/

clean:
	rm -f pt && rm -f pt_test && rm -f $(OBJDIR)/*.o

test: lib/pt.o lib/test.o
	$(CC) $(LD_FLAGS) -o pt_test lib/test.o lib/pt.o

#for debug reasons.
print-%  : ; @echo $* = $($*)
