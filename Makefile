SRCDIR := src
OBJDIR := lib
INCDIR := include
IFLAGS := -I$(INCDIR)/

CPP_FILES := $(wildcard $(SRCDIR)/*.cpp)
HPP_FILES := $(INCDIR)/*.hpp

LD_FLAGS :=
CC_FLAGS :=-std=c++14 -Wall -pedantic -O3

all: pt doc

debug: CC_FLAGS += -g
debug: LD_FLAGS += -g
debug: pt

pt : lib/main.o lib/pt.o
	g++ $(LD_FLAGS) -o $@ $^

lib/%.o : src/%.cpp $(HPP_FILES)
	g++ $(CC_FLAGS) $(IFLAGS) -c -o $@ $<

doc: .Doxyfile $(CPP_FILES) $(HPP_FILES) 
	doxygen $^

doc_clean:
	rm -r doc/html/ && rm -r doc/latex/

clean:
	rm -f pt && rm -f pt_test && rm -f $(OBJDIR)/*.o

test: lib/pt.o lib/test.o
	g++ $(LD_FLAGS) -o pt_test lib/test.o lib/pt.o

#for debug reasons.
print-%  : ; @echo $* = $($*)
