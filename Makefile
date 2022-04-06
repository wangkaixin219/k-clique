OBJ = main.o data.o k-clique.o
CC = gcc
CXX = g++
CFLAGS = -std=c++2a -Wall -O3 -g


all: $(OBJ)
	$(CXX) $(CFLAGS) -o k-clique $(OBJ)

main.o: main.cpp defs.h 
	$(CXX) $(CFLAGS) -c main.cpp

data.o: data.cpp defs.h
	$(CXX) $(CFLAGS) -c data.cpp

k-clique.o: k-clique.cpp  defs.h
	$(CXX) $(CFLAGS) -c k-clique.cpp


.PHONY: clean
clean:
	rm k-clique $(OBJ)
