#include "defs.h"

#define GF "../data/syn.edges"
#define OF "../data/syn.order"

int main(int argc, const char* argv[]) {

    graph g;
    syn_graph(g, 1000, 0.5);

    save_graph(g, GF);
    save_order(g, OF);

    dag(g, OF);
    set<unsigned> res;
    k_clique(g, res, 4);
}
