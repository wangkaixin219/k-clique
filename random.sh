#!/bin/zsh

make

./k-clique -g 500000 0.4
echo

./k-clique -r syn.edges -k 6 -o degeneracy
./k-clique -r syn.edges -k 6 -o degree

for k in {1..100}
do 
    ./k-clique -r syn.edges -k 6 -o random
done


