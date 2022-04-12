#!/bin/zsh

find . -name "*.h" | xargs -n1 -I{} docker cp {} ubuntu:/root/k-clique
find . -name "*.cpp" | xargs -n1 -I{} docker cp {} ubuntu:/root/k-clique
find . -name "*.sh" | xargs -n1 -I{} docker cp {} ubuntu:/root/k-clique
find ./data -name "*.edges" | xargs -n1 -I{} docker cp {} ubuntu:/root/k-clique/data
