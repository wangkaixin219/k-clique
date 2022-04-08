
#include "defs.h"

void read_graph(graph& g, const string& graph_file) {
    ifstream f(graph_file);
    string line;

    while (getline(f, line)) {
        if (line[0] == '%') continue;
        unsigned u, v;
        stringstream ss(line);
        ss >> u >> v;
        g.adj[u].insert(v);
        g.adj[v].insert(u);
    }

    f.close();
}

void syn_graph(graph& g, unsigned n, double p) {
    unsigned e = 0;

    for (unsigned i = 2; i <= n; ++i) {
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
            if (g.adj[u].size() <= n_link) {
                for (unsigned v : g.adj[u]) {
                    if (seen.count(v) == 0) {
                        q.push(v);
                        seen.insert(v);
                    }

                }
            }
            else {
                while (n_link--) {
                    unsigned v = rand_select(g.adj[u]);
                    if (seen.count(v) == 0) {
                        q.push(v);
                        seen.insert(v);
                    }
                }
            }
            g.adj[i].insert(u);
            g.adj[u].insert(i);
            e++;
            q.pop();
        }
        seen.clear();
    }
    cout << "|V| = " << n << ", |E| = " << e << endl;
}

void save_graph(const graph& g, const string& graph_file) {
    ofstream f(graph_file);
    for (pair<unsigned, set<unsigned>> u_adj : g.adj) {
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
    cout << "Adj ";
    for (pair<unsigned, set<unsigned>> u_adj : g.adj) {
        unsigned u = u_adj.first;
        set<unsigned> adj = u_adj.second;
        for (unsigned v : adj) {
            cout << "(" << u << ", " << v << ") ";
        }
    }
    cout << endl << "Color ";

    for (pair<unsigned, unsigned> u_c : g.color) {
        cout << "(" << u_c.first << ", " << u_c.second << ") ";
    }
    cout << endl << "Order ";

    for (pair<unsigned, unsigned> u_c : g.order) {
        cout << "(" << u_c.first << ", " << u_c.second << ") ";
    }
    cout << endl;
}

void print_result(const result& res) {
    printf("Runtime %8.2lf, calls %8u, cliques %8u\n\n", res.runtime, res.calls, res.cliques);
}
