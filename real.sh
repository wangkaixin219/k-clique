#!/bin/zsh

make

name="cleaned_cit-DBLP.edges"

for k in {3..10}
do 
    echo "k = $k"
    ./k-clique -r $name -k $k -o degeneracy
    ./k-clique -r $name -k $k -o degree
    ./k-clique -r $name -k $k -o random
    ./k-clique -r $name -k $k -o natrual
done


