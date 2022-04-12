#include "defs.h"

result_t results;
param_t parameters;

int main(int argc, const char* argv[]) {

    rusage start, end;
    arg_parser(argc, argv, &parameters);
    graph_t* g;

    if (parameters.n != 0) {
        g = forest_fire(parameters.n, parameters.p);
        fprintf(stdout, "\n|V| = %u, |E| = %u\n", g->N, g->M);
        write_graph(g, parameters.path);
        free_graph(g);
    }
    else {
        g = read_graph(parameters.path);
        ordering(g, parameters.order);
        GetCurTime(&start);
        k_clique(g, parameters.k);
        GetCurTime(&end);
        results.runtime = GetTime(&start, &end);
        free_graph(g);
        print_result(results);
    }
    return 0;
}