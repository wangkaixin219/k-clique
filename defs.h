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
#include <sys/resource.h>

using namespace std;

typedef struct {
    unsigned id, ord;
} vertex;


typedef map<unsigned, set<unsigned>> graph;


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

        // data.cpp
void read_graph(graph& g, const string& graph_file);
void syn_graph(graph& g, unsigned n, double p);
void save_graph(const graph& g, const string& graph_file);
void print_graph(const graph& g);

        // k-clique.cpp
void save_order(graph& g, const string& order_file);
void dag(graph& g, const string& order_file);
graph subgraph(graph& g, unsigned id);
void k_clique(graph g, set<unsigned>& res, unsigned l);


#endif //K_CLIQUE_DEFS_H
