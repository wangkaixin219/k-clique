
all:
	make -C C++/ all
	cp C++/k-clique Python/

clean:
	rm Python/k-clique
	rm -rf Python/__pycache__
	make -C C++/ clean
