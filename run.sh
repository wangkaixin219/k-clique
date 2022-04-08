#!/bin/zsh

make

./k-clique -g 10000 0.5
echo

for k in {3..10}
do 
    echo "k = $k"
    ./k-clique -r syn.edges -k $k -o degeneracy
    ./k-clique -r syn.edges -k $k -o degree
    ./k-clique -r syn.edges -k $k -o random
    ./k-clique -r syn.edges -k $k -o natrual
done

