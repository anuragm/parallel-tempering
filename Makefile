SRCDIR=src
IFLAGS=
LFLAGS=
CFLAGS=-std=c++14 -Wall

all: $(SRCDIR)/main.cpp
	g++ $(CFLAGS) $(LFLAGS) $(IFLAGS) -o pt $(SRCDIR)/main.cpp

clean:
	rm -f pt
