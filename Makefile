OBJ = main.o k-clique.o utils.o graph.o
CC = gcc
CXX = g++
CFLAGS = -std=c++2a -Wall -O3 -g


all: $(OBJ)
	$(CXX) $(CFLAGS) -o k-clique $(OBJ)

main.o: main.cpp defs.h 
	$(CXX) $(CFLAGS) -c main.cpp

k-clique.o: k-clique.cpp  defs.h
	$(CXX) $(CFLAGS) -c k-clique.cpp

utils.o: utils.cpp defs.h
	$(CXX) $(CFLAGS) -c utils.cpp

graph.o: graph.cpp defs.h
	$(CXX) $(CFLAGS) -c graph.cpp

.PHONY: clean
clean:
	rm k-clique $(OBJ)
