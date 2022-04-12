#ifndef K_CLIQUE_DEFS_H
#define K_CLIQUE_DEFS_H

#include <sys/resource.h>
#include <cstring>
#include <assert.h>
#include <random>

#define DEGENERACY      1
#define DEGREE          2
#define RANDOM          3
#define LEARNED         4
#define LEXICOGRAPHIC   5

#define MAX_E       100000      // define for malloc
#define MAX_V       100000      // define for malloc
#define MAX_DEG     256         // define for malloc

using namespace std;

/*
 * vertex ranges from 1 to n
 * edge stores in a list, e = (u, v) direction from u to v
 * at the beginning, the direction is not the final result
 */

typedef struct {
    unsigned s, t;  // s --> t
} edge_t;

typedef struct {
    unsigned idx;
    unsigned deg;
    unsigned* adj;
    unsigned depth;
} vertex_t;

typedef struct {
    unsigned N, M, P, D;
    edge_t* E;
    vertex_t* V;
    int* pos;    // map idx to the position id in V
} graph_t;

typedef struct {
    unsigned act_size;
    unsigned* act;
} mask_t;

typedef struct {
    double runtime;
    unsigned calls, cliques;
} result_t;

typedef struct {
    unsigned order, k, n;
    double p;
    char path[256];
} param_t;


typedef struct {
    unsigned* l;
    unsigned head, tail;
    unsigned max_size;
} queue_t;


typedef struct {
    queue_t* vq;
    bool* seen;
} ff_t;

typedef struct {
    unsigned key;
    double value;
} pair_t;

typedef struct {
    pair_t* l;
    unsigned* pos;      // for update
    unsigned cur_size;
    unsigned max_size;
} heap_t;

inline unsigned max3(unsigned a, unsigned b, unsigned c) {
    a = (a > b) ? a : b;
    return (a > c) ? a : c;
}

inline unsigned max2(unsigned a, unsigned b) {
    return (a > b) ? a : b;
}

inline int rand_int(int l, int u) {
    random_device rd;
    mt19937_64 gen(rd());
    uniform_int_distribution<> dist(l, u);
    return dist(gen);
}

inline double rand_real(double l, double u) {
    random_device rd;
    mt19937_64 gen(rd());
    uniform_real_distribution<> dist(l, u);
    return dist(gen);
}

inline long rand_geo(double p) {
    random_device rd;
    mt19937_64 gen(rd());
    geometric_distribution<long> dist(p);
    return dist(gen);
}

inline void GetCurTime(struct rusage* curTime) {
    if (getrusage(RUSAGE_SELF, curTime) != 0) {
        fprintf(stderr, "The running time info couldn't be collected successfully.\n");
        exit(0);
    }
}

inline double GetTime(struct rusage* start, struct rusage* end) {  // unit: ms
    return ((float)(end->ru_utime.tv_sec - start->ru_utime.tv_sec)) * 1e3 +
           ((float)(end->ru_utime.tv_usec - start->ru_utime.tv_usec)) * 1e-3;
}

        // graph.c
graph_t* forest_fire(unsigned n, double p);
void add_edge(graph_t* g, unsigned u, unsigned v);        // add edge to g->E
void add_vertex(graph_t* g, unsigned idx);              // add vertex to V, update pos
void add_neighbor(graph_t* g, unsigned u, unsigned v);               // update *.deg and *.adj, u -- v
void add_direct_neighbor(graph_t* g, unsigned s, unsigned t);         // update *.deg and *.adj, s --> t
unsigned degree(graph_t* g, unsigned idx);
unsigned* adj(graph_t* g, unsigned idx);
void clear(graph_t* g);
void free_graph(graph_t* g);
graph_t* read_graph(const char* path);
void write_graph(const graph_t* g, const char* path);

        // k-clique.c
void ordering(graph_t* g, unsigned type);
void k_clique(graph_t* g, unsigned l);

        //utils.c
queue_t* construct_queue();
void free_queue(queue_t* q);
void push(queue_t* q, unsigned v);
void pop(queue_t* q);
unsigned front(queue_t* q);
bool empty(queue_t* q);
void print_result(const result_t r);
void print_progress(unsigned finished, unsigned total);
void arg_parser(int argc, const char* argv[], param_t* parameters);
heap_t* construct_heap(unsigned n);
void free_heap(heap_t* h);
void insert(heap_t* h, unsigned key, double value);
void update(heap_t* h, unsigned key);
void pop(heap_t* h);
pair_t min_element(heap_t* h);

#endif //K_CLIQUE_DEFS_H

