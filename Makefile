SRCDIR=src
OBJDIR=lib
IFLAGS=-I$(SRCDIR)/

CPP_FILES := $(wildcard $(SRCDIR)/*.cpp)
OBJ_FILES := $(addprefix $(OBJDIR)/,$(notdir $(CPP_FILES:.cpp=.o)))
LD_FLAGS :=
CC_FLAGS :=-std=c++14 -Wall -pedantic

all: pt doc

debug: CC_FLAGS += -g
debug: LD_FLAGS += -g
debug: pt

pt : $(OBJ_FILES)
	g++ $(LD_FLAGS) -o $@ $^

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	g++ $(CC_FLAGS) $(IFLAGS) -c -o $@ $<

doc: $(CPP_FILES)
	doxygen .Doxyfile

doc_clean:
	rm -r doc/html/ && rm -r doc/latex/

clean:
	rm -f pt && rm -f $(OBJDIR)/*.o
