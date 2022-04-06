
#include "defs.h"

void read_graph(graph& g, const string& graph_file) {
    ifstream f(graph_file);
    string line;

    while (getline(f, line)) {
        if (line[0] == '%') continue;
        unsigned u, v;
        stringstream ss(line);
        ss >> u >> v;
        g[u].insert(v);
        g[v].insert(u);
    }

    f.close();
}

unsigned random_select(set<unsigned> candidate) {
    unsigned n = rand_int(0, candidate.size()-1);
    set<unsigned>::iterator it = candidate.begin();
    advance(it, n);
    return *(it);
}

void syn_graph(graph& g, unsigned n, double p) {

    for (unsigned i = 2; i < n; ++i) {
        fprintf(stdout, "Generate %.2lf%% edges\r", (double) i / n * 100);
        fflush(stdout);
        unsigned ambassador = rand_int(1, i-1);
        queue<unsigned> q;
        set<unsigned> seen;
        q.push(ambassador);
        seen.insert(i);
        seen.insert(ambassador);

        while (!q.empty()) {
            long u = q.front();
            unsigned n_link = rand_geo(p);
            if (g[u].size() <= n_link) {
                for (unsigned v : g[u]) {
                    if (seen.count(v) == 0) {
                        q.push(v);
                        seen.insert(v);
                    }

                }
            }
            else {
                while (n_link--) {
                    unsigned v = random_select(g[u]);
                    if (seen.count(v) == 0) {
                        q.push(v);
                        seen.insert(v);
                    }
                }
            }
            g[i].insert(u);
            g[u].insert(i);
            q.pop();
        }
        seen.clear();
    }
}

void save_graph(const graph& g, const string& graph_file) {
    ofstream f(graph_file);
    for (pair<unsigned, set<unsigned>> u_adj : g) {
        unsigned u = u_adj.first;
        set<unsigned> adj = u_adj.second;
        for (unsigned v : adj) {
            if (u < v)
                f << u << " " << v << endl;
        }
    }
    f.close();
}

void print_graph(const graph& g) {
    for (pair<unsigned, set<unsigned>> u_adj : g) {
        unsigned u = u_adj.first;
        set<unsigned> adj = u_adj.second;
        for (unsigned v : adj) {
            cout << "(" << u << ", " << v << ") ";
        }
    }
    cout << endl;
}