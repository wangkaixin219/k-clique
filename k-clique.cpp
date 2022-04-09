#include "defs.h"

extern result results;

void degeneracy_ordering(graph& g) {
    map<unsigned, set<unsigned>> adj = g.adj;
    unsigned i = 1;
    while (!adj.empty()) {

        pair

        unsigned min_u = 0, min_deg = INT_MAX;
        for (pair<unsigned, set<unsigned>> u_adj : adj) {
            unsigned u = u_adj.first, deg = adj[u].size();
            if (deg < min_deg) {
                min_deg = deg;
                min_u = u;
            }
        }

        for (unsigned v : adj[min_u]) {
            adj[v].erase(min_u);
        }
        adj.erase(min_u);
        g.order[min_u] = i++;
    }
}

void degree_ordering(graph& g) {
    vector<pair<unsigned, unsigned>> V;
    for (pair<unsigned, set<unsigned>> u_adj : g.adj) {
        unsigned u = u_adj.first;
        V.push_back(make_pair(g.adj[u].size(), u));
    }
    sort(V.begin(), V.end());
    for(unsigned i = 0; i < V.size(); i++) {
        g.order[V[i].second] = i + 1;
    }
}

void random_ordering(graph& g) {
    set<unsigned> unused;
    for(unsigned i = 1; i <= g.adj.size(); i++) unused.insert(i);
    for(pair<unsigned, set<unsigned>> u_adj : g.adj) {
        unsigned u = u_adj.first, r = rand_select(unused);
        g.order[u] = r;
        unused.erase(r);
    }
}

void lexicographic_ordering(graph& g) {
    for (pair<unsigned, set<unsigned>> u_adj : g.adj) {
        unsigned u = u_adj.first;
        set<unsigned> adj = u_adj.second;
        if (!g.order.contains(u)) g.order[u] = u;
        for (unsigned v : adj) {
            if (!g.order.contains(v)) g.order[v] = v;
        }
    }
}


void color(graph& g) {
    map<unsigned, unsigned> reverse_raw_order;
    map<unsigned, set<unsigned>> reverse_color;

    for (pair<unsigned, unsigned> u_r : g.order) {
        unsigned u = u_r.first, r = u_r.second;
        reverse_raw_order[r] = u;
    }
    for (unsigned i = 1; reverse_raw_order.contains(i); i++) {
        unsigned u = reverse_raw_order[i], c = 1;
        set<unsigned> used;

        for (unsigned v : g.adj[u]) {
            if (g.color.contains(v)) used.insert(g.color[v]);
        }

        while (used.contains(c)) c++;
        g.color[u] = c;
        reverse_color[c].insert(u);
    }
    g.order.clear();
    unsigned i = 1;
    for (unsigned c = 1; reverse_color.contains(c); c++) {
        for (unsigned u : reverse_color[c]) {
            g.order[u] = i++;
        }
    }
}

void order(graph& g, unsigned type) {
    if (type == DEGENERACY) {
        printf("Degeneracy based ordering\n");
        degeneracy_ordering(g);
    }
    else if (type == DEGREE) {
        printf("Degree based ordering\n");
        degree_ordering(g);
    }
    else if (type == RANDOM) {
        printf("Random based ordering\n");
        random_ordering(g);
    }
    else if (type == LEARNED) {
        // TODO
    }
    else {
        printf("Lexicographic based ordering\n");
        lexicographic_ordering(g);
    }

    color(g);
}



void dag(graph& g) {
    set<unsigned> empty;
    for (pair<unsigned, set<unsigned>> u_adj : g.adj) {
        unsigned u = u_adj.first;
        set<unsigned> adj = u_adj.second;
        if (adj.empty()) empty.insert(u);

        for (unsigned v : adj) {
            if (g.order[u] > g.order[v]) {
                g.adj[v].erase(u);
            }
        }
    }

    for (unsigned u : empty)
        g.adj.erase(u);
}

graph subgraph(graph& g, unsigned id) {
    graph sub;
    sub.order = g.order;
    sub.color = g.color;

    for (unsigned v : g.adj[id]) {
        set_intersection(g.adj[v].begin(), g.adj[v].end(), g.adj[id].begin(), g.adj[id].end(),
                         inserter(sub.adj[v], sub.adj[v].begin()));
    }

    return sub;
}

void output(set<unsigned>& res, unsigned u, unsigned v) {
    cout << "Find clique {";
    for (unsigned r : res)
        cout << r << " ";
    cout << u << " " << v << "}." << endl;
}


void k_clique(graph g, set<unsigned>& res, unsigned l) {
    results.calls++;
    if (l == 2) {
        for (pair<unsigned, set<unsigned>> u_adj : g.adj) {
            unsigned u = u_adj.first;
            set<unsigned> adj = u_adj.second;
            for (unsigned v : adj) {
//                output(res, u, v);
                results.cliques++;
            }
        }
    }
    else {
        for (pair<unsigned, set<unsigned>> u_adj : g.adj) {
            unsigned u = u_adj.first;
            if (g.color[u] < l) continue;
            res.insert(u);
            k_clique(subgraph(g, u), res, l - 1);
            res.erase(u);
        }
    }
}
