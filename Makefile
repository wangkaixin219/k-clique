
all:
	make -C C++/ all
	cp C++/k-clique Python/

clean:
	rm Python/k-clique
	make -C C++/ clean
