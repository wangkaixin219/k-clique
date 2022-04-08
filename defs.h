//
// Created by Wang Kaixin on 4/6/22.
//

#ifndef K_CLIQUE_DEFS_H
#define K_CLIQUE_DEFS_H

#include <vector>
#include <map>
#include <set>
#include <queue>
#include <random>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iostream>
#include <climits>
#include <sys/resource.h>

#define DEGENERACY      1
#define DEGREE          2
#define RANDOM          3
#define LEARNED         4
#define LEXICOGRAPHIC   5

using namespace std;

typedef struct {
    unsigned order;     // global order
    unsigned k;         // global k
    unsigned n;         // syn n
    double p;           // syn p
    string path;        // global path
} param;

typedef struct {
    map<unsigned, set<unsigned>> adj;
    map<unsigned, unsigned> color, order;
} graph;


typedef struct {
    double runtime;
    unsigned calls, cliques;
} result;


inline int rand_int(int l, int u) {
    random_device rd;
    mt19937_64 gen(rd());
    uniform_int_distribution<> dist(l, u);
    return dist(gen);
}

inline unsigned rand_select(set<unsigned> candidate) {
    unsigned n = rand_int(0, candidate.size()-1);
    set<unsigned>::iterator it = candidate.begin();
    advance(it, n);
    return *(it);
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

        // parser.cpp
void arg_parser(int argc, const char* argv[], param& parameters);

        // data.cpp
void read_graph(graph& g, const string& graph_file);
void syn_graph(graph& g, unsigned n, double p);
void save_graph(const graph& g, const string& graph_file);
void print_graph(const graph& g);
void print_result(const result& res);

        // k-clique.cpp
void order(graph& g, unsigned type);
void dag(graph& g);
graph subgraph(graph& g, unsigned id);
void k_clique(graph g, set<unsigned>& res, unsigned l);


#endif //K_CLIQUE_DEFS_H
