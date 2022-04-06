

#include "defs.h"


void save_order(graph& g, const string& order_file) {
    map<unsigned, unsigned> order;
#ifdef DEGENERACY
    printf("Degeneracy based ordering\n");
    graph tg = g;
    unsigned i = 1;
    while (!tg.empty()) {
        unsigned min_u = 0, min_deg = INT_MAX;
        for (pair<unsigned, set<unsigned>> u_adj : tg) {
            unsigned u = u_adj.first, deg = g[u].size();
            if (deg < min_deg) {
                min_deg = deg;
                min_u = u;
            }
        }

        for (unsigned v : tg[min_u]) {
            tg[v].erase(min_u);
        }
        tg.erase(min_u);
        order[min_u] = i++;
    }
#elifdef DEGREE
    printf("Degree based ordering\n");
    vector<pair<unsigned, unsigned>> V;
    for (pair<unsigned, set<unsigned>> u_adj : g) {
        unsigned u = u_adj.first;
        V.push_back(make_pair(g[u].size(), u));
    }
    sort(V.begin(), V.end());
    for(unsigned i = 0; i < V.size(); i++) {
        order[V[i].second] = i + 1;
    }
#elifdef RANDOM
    // TODO
#elifdef LEARNED
    // TODO
#else
    printf("Lexicographic based ordering\n");
    for (pair<unsigned, set<unsigned>> u_adj : g) {
        unsigned u = u_adj.first;
        set<unsigned> adj = u_adj.second;
        if (!order.contains(u)) order[u] = u;
        for (unsigned v : adj) {
            if (!order.contains(v)) order[v] = v;
        }
    }
#endif
    ofstream f(order_file);
    for (pair<unsigned, unsigned> p : order) {
        f << p.first << " " << p.second << endl;
    }
    f.close();
}



void dag(graph& g, const string& order_file) {
    ifstream f(order_file);
    string line;
    map<unsigned, unsigned> order;

    while(getline(f, line)) {
        stringstream ss(line);
        unsigned u, r;
        ss >> u >> r;
        order[u] = r;
    }

    for (pair<unsigned, set<unsigned>> u_adj : g) {
        unsigned u = u_adj.first;
        set<unsigned> adj = u_adj.second;
        for (unsigned v : adj) {
            if (order[u] < order[v]) {
                g[v].erase(u);
            }
        }
    }
    f.close();
}

graph subgraph(graph& g, unsigned id) {
    graph sub_g;

    for (unsigned v : g[id]) {
        set_intersection(g[v].begin(), g[v].end(), g[id].begin(), g[id].end(),
                         inserter(sub_g[v], sub_g[v].begin()));
    }
    return sub_g;
}

void output(set<unsigned>& res, unsigned u, unsigned v) {
    cout << "Find clique {";
    for (unsigned r : res)
        cout << r << " ";
    cout << u << " " << v << "}." << endl;
}


void k_clique(graph g, set<unsigned>& res, unsigned l) {
    if (l == 2) {
        for (pair<unsigned, set<unsigned>> u_adj : g) {
            unsigned u = u_adj.first;
            set<unsigned> adj = u_adj.second;
            for (unsigned v : adj) {
                output(res, u, v);
            }
        }
    }
    else {
        for (pair<unsigned, set<unsigned>> u_adj : g) {
            unsigned u = u_adj.first;
            res.insert(u);
            k_clique(subgraph(g, u), res, l - 1);
            res.erase(u);
        }
    }
}
