#include "defs.h"

result results;

int main(int argc, const char* argv[]) {

    param parameters;
    rusage start, end;
    arg_parser(argc, argv, parameters);

    if (parameters.n != 0) {        // generate graph
        graph g;
        syn_graph(g, parameters.n, parameters.p);
        save_graph(g, parameters.path);
        exit(0);
    }

    graph G;
    set<unsigned> R;
    read_graph(G, parameters.path);
    order(G, parameters.order);
    dag(G);

    GetCurTime(&start);
    k_clique(G, R, parameters.k);
    GetCurTime(&end);
    results.runtime = GetTime(&start, &end);
    print_result(results);


}
