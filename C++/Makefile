SRC := $(shell find . -name "*.cpp")
INC := $(shell find . -name "*.h")
OBJ := $(SRC:.cpp=.o)
CXX := g++
CFLAGS := -std=c++11 -Wall -O3 -g


all: $(OBJ)
	$(CXX) $(CFLAGS) -o k-clique $(OBJ)

%.o: %.cpp $(INC)
	$(CXX) $(CFLAGS) -c $< -o $@


.PHONY: clean
clean:
	rm k-clique $(OBJ)
