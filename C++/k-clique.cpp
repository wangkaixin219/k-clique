#include "defs.h"

extern result_t results;
extern param_t parameters;
unsigned *color, *order;
mask_t** mask;

void free_all() {
    if (order) {
        free(order);
        order = nullptr;
    }
    if (color) {
        free(color);
        color = nullptr;
    }
    if (mask) {
        for (int i = 2; i <= parameters.k; ++i) {
            if (mask[i]) {
                free(mask[i]->act);
                mask[i]->act = nullptr;
                free(mask[i]);
                mask[i] = nullptr;
            }
        }
        free(mask);
        mask = nullptr;
    }
}

void degeneracy_ordering(graph_t* g) {
    order = (unsigned*) calloc((g->P + 1), sizeof(unsigned));
    heap_t* h = construct_heap(g->P + 1);
    for (int i = 0; i < g->N; ++i) insert(h, g->V[i].idx, -g->V[i].deg);


    for (int i = 1; i <= g->N; ++i) {
        unsigned u = min_element(h).key;
        order[u] = i;
        pop(h);
        for (int j = 0; j < degree(g, u); ++j) {
            unsigned v = adj(g, u)[j];
            update(h, v);
        }
    }

    free_heap(h);
}

void degree_ordering(graph_t* g) {
    order = (unsigned*) calloc((g->P + 1), sizeof(unsigned));
    heap_t* h = construct_heap(g->P + 1);
    for (int i = 0; i < g->N; ++i) insert(h, g->V[i].idx, -g->V[i].deg);

    for (int i = 1; i <= g->N; ++i) {
        unsigned u = min_element(h).key;
        order[u] = i;
        pop(h);
    }
    free_heap(h);
}

void random_ordering(graph_t* g) {
    order = (unsigned*) calloc((g->P + 1), sizeof(unsigned));
    heap_t* h = construct_heap(g->P + 1);
    random_device rd;
    seed_seq sd{rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd()};
    mt19937_64 gen(sd);
    uniform_real_distribution<> dist(0, 1);

    for (int i = 0; i < g->N; ++i) insert(h, g->V[i].idx,  dist(gen));

    for (int i = 1; i <= g->N; ++i) {
        unsigned u = min_element(h).key;
        order[u] = i;
        pop(h);
    }
    free_heap(h);
}

void lexicographic_ordering(graph_t* g) {
    order = (unsigned*) calloc((g->P + 1), sizeof(unsigned));
    heap_t* h = construct_heap(g->P + 1);
    for (int i = 0; i < g->N; ++i) insert(h, g->V[i].idx, -g->V[i].idx);

    for (int i = 1; i <= g->N; ++i) {
        unsigned u = min_element(h).key;
        order[u] = i;
        pop(h);
    }

    free_heap(h);
}

void learned_ordering(graph_t* g) {

    FILE* fp = fopen(parameters.order_path, "r");
    order = (unsigned*) calloc((g->P + 1), sizeof(unsigned));
    unsigned u, i = 1;
    while (fscanf(fp, "%u", &u) == 1) {
        order[u] = i++;
    }
    fclose(fp);
}

void coloring(graph_t* g) {
    color = (unsigned*) calloc((g->P+1), sizeof(unsigned));
    bool* used = (bool*) calloc((g->D+1), sizeof(bool));
    unsigned* reverse_order = (unsigned*) malloc((g->N + 1) * sizeof(unsigned));
    unsigned max_color = 1;

    for (int i = 1; i <= g->P; ++i) if (order[i]) reverse_order[order[i]] = i;

    for (int i = 1; i <= g->N; ++i) {
        unsigned u = reverse_order[i];
        for (int j = 0; j < degree(g, u); ++j) {
            unsigned v = adj(g, u)[j];
            if (color[v])
                used[color[v]] = true;
        }

        unsigned c = 1;
        while (used[c]) c++;
        color[u] = c;
        max_color = c > max_color ? c : max_color;

        for (int j = 0; j < degree(g, u); ++j) {
            unsigned v = adj(g, u)[j];
            if (color[v])
                used[color[v]] = false;
        }
    }

    clear(g);
    g->D = 0;
    for (int i = 0; i < g->M; ++i) {
        unsigned s = g->E[i].s, t = g->E[i].t;

        if (color[s] > color[t]) add_direct_neighbor(g, s, t);
        else if (color[s] < color[t]) add_direct_neighbor(g, t, s);
        else printf("Two vertices of edge (%u %u) have same color.\n", s, t);

        g->D = max3(g->D, degree(g, s), degree(g, t));
    }

    printf("Max degree = %u after coloring (used %u colors)\n", g->D, max_color);

    free(used);
    free(reverse_order);
}




void ordering(graph_t* g, unsigned type) {
    if (type == DEGENERACY) {
        printf("Degeneracy ordering\n");
        degeneracy_ordering(g);
        printf("Degeneracy ordering finish\n");
    }
    else if (type == DEGREE) {
        printf("Degree ordering\n");
        degree_ordering(g);
        printf("Degree ordering finish\n");
    }
    else if (type == RANDOM) {
        printf("Random ordering\n");
        random_ordering(g);
        printf("Random ordering finish\n");
    }
    else if (type == LEARNED) {
        printf("Learned ordering\n");
        learned_ordering(g);
        printf("Learned ordering finish\n");
    }
    else {
        printf("Lexicographic ordering\n");
        lexicographic_ordering(g);
        printf("Lexicographic ordering finish\n");
    }

    coloring(g);
}


void k_clique(graph_t* g, unsigned l) {
    results.calls++;

    if (l == parameters.k) {
        mask = (mask_t**) malloc( (l + 1) * sizeof(mask_t*));
        for (int i = 2; i < l; ++i) {
            mask[i] = (mask_t*) malloc(sizeof(mask_t));
            mask[i]->act = (unsigned*) malloc(g->D * sizeof(unsigned));
        }
        mask[l] = (mask_t*) malloc(sizeof(mask_t));
        mask[l]->act_size = g->N;
        mask[l]->act = (unsigned*) malloc(g->N * sizeof(unsigned));
        for (int i = 0; i < g->N; ++i) mask[l]->act[i] = g->V[i].idx;
    }

    if (l == 2) {

        for (int i = 0; i < mask[l]->act_size; ++i) {
            unsigned u = mask[l]->act[i];
            for (int j = 0; j < degree(g, u); ++j) {
                unsigned v = adj(g, u)[j];
                if (g->V[g->pos[v]].depth == l) {
                    results.cliques++;
                }
            }
        }
    }
    else {
        for (int i = 0; i < mask[l]->act_size; ++i) {
            unsigned u = mask[l]->act[i];

            if (color[u] < l) continue;

            mask[l - 1]->act_size = 0;
            for (int j = 0; j < degree(g, u); ++j) {
                unsigned v = adj(g, u)[j];
                if (g->V[g->pos[v]].depth == l) {
                    mask[l - 1]->act[mask[l - 1]->act_size++] = v;
                    g->V[g->pos[v]].depth--;
                }
            }

            k_clique(g, l - 1);

            for (int j = 0; j < mask[l - 1]->act_size; ++j) {
                unsigned v = mask[l - 1]->act[j];
                g->V[g->pos[v]].depth++;
            }
        }
    }

    if (l == parameters.k) free_all();
}





