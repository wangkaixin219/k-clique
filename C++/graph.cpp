#include "defs.h"
#include <assert.h>

extern param_t parameters;

ff_t* construct_ff(unsigned n) {
    ff_t* ff = (ff_t*) malloc(sizeof(ff_t));
    ff->seen = (bool*) calloc(n+1, sizeof(bool));
    ff-> vq = construct_queue();
    return ff;
}

void reset_ff(ff_t* ff) {
    for (unsigned k = 0; k < ff->vq->tail; ++k) {
        ff->seen[ff->vq->l[k]] = false;
    }
    ff->vq->head = ff->vq->tail = 0;
}

void free_ff(ff_t* ff) {
    free_queue(ff->vq);
    free(ff->seen);
}

void clear(graph_t* g) {
    for (unsigned i = 0; i < g->N; ++i) {
        if (g->V[i].deg) {
            free(g->V[i].adj);
            g->V[i].adj = nullptr;
            g->V[i].deg = 0;
        }
    }
}

void add_direct_neighbor(graph_t* g, unsigned s, unsigned t) {
    if (g->V[g->pos[s]].deg == 0) g->V[g->pos[s]].adj = (unsigned*) malloc(MAX_DEG * sizeof(unsigned ));
    else if (g->V[g->pos[s]].deg % MAX_DEG == 0) g->V[g->pos[s]].adj = (unsigned*) realloc(g->V[g->pos[s]].adj, (g->V[g->pos[s]].deg + MAX_DEG) * sizeof(unsigned));
    assert(g->V[g->pos[s]].idx == s);
    assert( g->V[g->pos[t]].idx == t);
    g->V[g->pos[s]].adj[g->V[g->pos[s]].deg++] =  g->V[g->pos[t]].idx;
}

void add_neighbor(graph_t* g, unsigned u, unsigned v) {
    add_direct_neighbor(g, u, v);
    add_direct_neighbor(g, v, u);
}

void add_edge(graph_t* g, unsigned u, unsigned v) {
    if (g->M == 0) g->E = (edge_t*) malloc(MAX_E * sizeof(edge_t));
    else if (g->M % MAX_E == 0) g->E = (edge_t*) realloc(g->E, (g->M + MAX_E) * sizeof(edge_t));

    g->E[g->M].s = u;
    g->E[g->M].t = v;
    g->M++;
}

unsigned degree(graph_t* g, unsigned idx) {
    if (g->pos[idx] < 0) {
        printf("degree(): vertex %u NOT exists\n", idx);
        return 0;
    }
    return g->V[g->pos[idx]].deg;
}

unsigned* adj(graph_t* g, unsigned idx) {
    if (g->pos[idx] < 0) {
        printf("adj(): vertex %u NOT exists\n", idx);
        return nullptr;
    }
    return g->V[g->pos[idx]].adj;
}

void add_vertex(graph_t* g, unsigned idx) {
    if (g->pos[idx] >= 0) return;
    else {
        if (g->N == 0) g->V = (vertex_t*) malloc(MAX_V * sizeof(vertex_t));
        else if (g->N % MAX_V == 0) g->V = (vertex_t*) realloc(g->V, (g->N + MAX_V) * sizeof(vertex_t));

        g->V[g->N].idx = idx;
        g->V[g->N].deg = 0;
        g->V[g->N].depth = parameters.k;
        g->V[g->N].adj = nullptr;
        g->pos[idx] = g->N;
        g->N++;
    }
}

graph_t* forest_fire(unsigned n, double p) {
    graph_t* g = (graph_t*) malloc(sizeof(graph_t));
    ff_t* ff = construct_ff(n);

    g->pos = (int*) malloc((n+1) * sizeof(int));
    memset(g->pos, -1, (n+1) * sizeof(int));
    g->N = g->M = 0;

    for (unsigned idx = 1; idx <= n; ++idx) {
        print_progress(idx, n);
        add_vertex(g, idx);

        if (g->N > 1) {
            unsigned s = rand_int(1, idx-1);
            push(ff->vq, s);
            ff->seen[idx] = ff->seen[s] = true;

            while (!empty(ff->vq)) {
                unsigned u = front(ff->vq), link = rand_geo(p);

                if (degree(g, u) <= link) {
                    for (unsigned j = 0; j < degree(g, u); ++j) {
                        unsigned v = adj(g, u)[j];
                        if (!ff->seen[v]) {
                            push(ff->vq, v);
                            ff->seen[v] = true;
                        }
                    }
                }
                else {
                    while (link--) {
                        unsigned v = adj(g, u)[rand_int(0, degree(g, u) - 1)];
                        if (!ff->seen[v]) {
                            push(ff->vq, v);
                            ff->seen[v] = true;
                        }
                    }
                }
                add_neighbor(g, u, idx);
                add_edge(g, u, idx);
                pop(ff->vq);
            }
            reset_ff(ff);
        }
    }

    free_ff(ff);
    return g;
}

void free_graph(graph_t* g) {
    if (g) {
        if (g->E) {
            free(g->E);
            g->E = nullptr;
        }
        if (g->V) {
            for (unsigned i = 0; i < g->N; ++i) {
                if (g->V[i].deg) {
                    free(g->V[i].adj);
                    g->V[i].adj = nullptr;
                }
            }
            free(g->V);
            g->V = nullptr;
        }
        if (g->pos) {
            free(g->pos);
            g->pos = nullptr;
        }

        free(g);
        g = nullptr;
    }
}

graph_t* read_graph(const char* path) {
    graph_t* g = (graph_t*) malloc(sizeof(graph_t));
    FILE *fp = fopen(path, "r");
    unsigned max_v = 0, s, t;

    g->D = g->N = g->M = g->P = 0;
    g->pos = (int*) malloc((g->P + 1) * sizeof(int));
    memset(g->pos, -1, (g->P + 1) * sizeof(int));

    while (fscanf(fp, "%u %u", &s, &t) == 2) {

        if ((max_v = max3(max_v, s, t)) > g->P) {
            g->pos = (int*) realloc(g->pos, (max_v + 1) * sizeof(int));
            memset(g->pos + g->P + 1, -1, (max_v - g->P) * sizeof(int));
            g->P = max_v;
        }

        add_vertex(g, s);
        add_vertex(g, t);
        add_neighbor(g, s, t);
        add_edge(g, s, t);

        g->D = max3(g->D, degree(g, s), degree(g, t));
    }

    g->E = (edge_t*) realloc(g->E, g->M * sizeof(edge_t));
    g->V = (vertex_t*) realloc(g->V, g->N * sizeof(vertex_t));

    fclose(fp);
    printf("|V| = %u, |E| = %u, Max index = %u, Max degree = %u\n", g->N, g->M, g->P, g->D);

    return g;
}

void write_graph(const graph_t* g, const char* path) {
    FILE* fp = fopen(path, "w");
    for (unsigned i = 0; i < g->M; ++i) {
        unsigned u = g->E[i].s, v = g->E[i].t;
        if (u < v) fprintf(fp, "%u %u\n", u, v);
        else fprintf(fp, "%u %u\n", v, u);
    }
    fclose(fp);
}
