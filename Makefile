
all:
	make -C C++/ all
	cp C++/k-clique Python/

clean:
	make -C Python/ clean
	make -C C++/ clean
