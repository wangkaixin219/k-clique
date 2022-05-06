#include "defs.h"

#define L   10000

queue_t* construct_queue() {
    queue_t* q = (queue_t*) malloc(sizeof(queue_t));
    q->l = (unsigned*) malloc(L * sizeof(unsigned ));
    q->head = q->tail = 0;
    q->max_size = L;
    return q;
}

void free_queue(queue_t* q) {
    free(q->l);     q->l = nullptr;
    free(q);        q = nullptr;
}

void push(queue_t* q, unsigned v) {
    q->l[q->tail++] = v;
    if (q->tail == q->max_size) {
        q->max_size += L;
        q->l = (unsigned*) realloc(q->l, q->max_size * sizeof(unsigned ));
    }
}

void pop(queue_t* q) {
    assert(!empty(q));
    q->head++;
}

unsigned front(queue_t* q) {
    assert(!empty(q));
    return q->l[q->head];
}

bool empty(queue_t* q) {
    return q->tail == q->head;
}


max_heap_t* construct_max_heap(unsigned n) {
    max_heap_t* h = (max_heap_t*) malloc(sizeof(max_heap_t));
    h->max_size = n;
    h->cur_size = 0;
    h->l = (pair_t*) malloc((n+1) * sizeof(pair_t));
    h->pos = (unsigned*) malloc((n+1) * sizeof(unsigned));
    for (int i = 0; i < n+1; ++i) h->pos[i] = -1;
    return h;
}

void free_max_heap(max_heap_t* h) {
    free(h->l);     h->l = nullptr;
    free(h->pos);   h->pos = nullptr;
    free(h);        h = nullptr;
}

inline unsigned parent(unsigned i) {
    assert(i >= 1);
    return (i - 1) >> 1;
}

inline unsigned left_child(unsigned i) {
    return (i << 1) + 1;
}

inline unsigned right_child(unsigned i) {
    return (i + 1) << 1;
}

void swap(max_heap_t* h, unsigned i, unsigned j) {
    pair_t tmp = h->l[i];
    h->l[i] = h->l[j];
    h->l[j] = tmp;

    h->pos[h->l[i].key] = i;
    h->pos[h->l[j].key] = j;
}

void bubble_up(max_heap_t* h, unsigned i) {
    while (i > 0 && h->l[i].value > h->l[parent(i)].value) {
        swap(h, i, parent(i));
        i = parent(i);
    }
}

void bubble_down(max_heap_t* h, unsigned i) {
    unsigned l = left_child(i), r = right_child(i);

    while (l < h->cur_size) {
        unsigned child = (r < h->cur_size) && (h->l[r].value > h->l[l].value) ? r : l;
        if (h->l[i].value < h->l[child].value) {
            swap(h, i, child);
            i = child, l = left_child(i), r = right_child(i);
            continue;
        }
        break;
    }
}

void insert(max_heap_t* h, unsigned key, double value) {
    if (h->cur_size == h->max_size) {
        h->max_size += L;
        h->l = (pair_t*) realloc(h->l,  h->max_size * sizeof(pair_t));
    }
    unsigned i = h->cur_size++;
    h->l[i].key = key, h->l[i].value = value;
    h->pos[key] = i;

    bubble_up(h, i);
}

void pop(max_heap_t* h) {
    h->pos[h->l[0].key] = -1;
    h->l[0] = h->l[--h->cur_size];
    h->pos[h->l[0].key] = 0;
    bubble_down(h, 0);
}

//void update(max_heap_t* h, unsigned key) {
//    unsigned i = h->pos[key];
//    if (i != -1) {
//        h->l[i].value--;
//        bubble_down(h, i);
//    }
//}

pair_t max_element(max_heap_t* h) {
    return h->l[0];
}

void swap(min_heap_t* h, unsigned i, unsigned j) {
    pair_t tmp = h->l[i];
    h->l[i] = h->l[j];
    h->l[j] = tmp;

    h->pos[h->l[i].key] = i;
    h->pos[h->l[j].key] = j;
}

void bubble_up(min_heap_t* h, unsigned i) {
    while (i > 0 && h->l[i].value < h->l[parent(i)].value) {
        swap(h, i, parent(i));
        i = parent(i);
    }
}

void bubble_down(min_heap_t* h, unsigned i) {
    unsigned l = left_child(i), r = right_child(i);

    while (l < h->cur_size) {
        unsigned child = (r < h->cur_size) && (h->l[r].value < h->l[l].value) ? r : l;
        if (h->l[i].value > h->l[child].value) {
            swap(h, i, child);
            i = child, l = left_child(i), r = right_child(i);
            continue;
        }
        break;
    }
}

min_heap_t* construct_min_heap(unsigned n) {
    min_heap_t* h = (min_heap_t*) malloc(sizeof(min_heap_t));
    h->max_size = n;
    h->cur_size = 0;
    h->l = (pair_t*) malloc((n+1) * sizeof(pair_t));
    h->pos = (unsigned*) malloc((n+1) * sizeof(unsigned));
    for (int i = 0; i < n+1; ++i) h->pos[i] = -1;
    return h;
}

void free_min_heap(min_heap_t* h) {
    free(h->l);     h->l = nullptr;
    free(h->pos);   h->pos = nullptr;
    free(h);        h = nullptr;
}

void insert(min_heap_t* h, unsigned key, double value) {
    if (h->cur_size == h->max_size) {
        h->max_size += L;
        h->l = (pair_t*) realloc(h->l,  h->max_size * sizeof(pair_t));
    }
    unsigned i = h->cur_size++;
    h->l[i].key = key, h->l[i].value = value;
    h->pos[key] = i;

    bubble_up(h, i);
}

void update(min_heap_t* h, unsigned key) {
    unsigned i = h->pos[key];
    if (i != -1) {
        h->l[i].value--;
        bubble_up(h, i);
    }
}

void pop(min_heap_t* h) {
    h->pos[h->l[0].key] = -1;
    h->l[0] = h->l[--h->cur_size];
    h->pos[h->l[0].key] = 0;
    bubble_down(h, 0);
}

pair_t min_element(min_heap_t* h) {
    return h->l[0];
}


void print_result(const result_t r) {
    printf("Runtime %8.2lf, calls %8u, cliques %8u\n\n", r.runtime, r.calls, r.cliques);
}

void print_progress(unsigned finished, unsigned total) {
    fprintf(stdout, "Finish %.2lf%%\r", (double) finished / total * 100);
    fflush(stdout);
}

void arg_parser(int argc, const char* argv[], param_t* parameters) {
    if (argc == 4) {
        if (strcmp(argv[1], "-g") == 0) {
            parameters->n = atoi(argv[2]);
            parameters->p = atof(argv[3]);
            strcpy(parameters->edge_path, "./data/syn.edges");
        }
        else {
            fprintf(stderr, "Usage error.\n");
            exit(0);
        }
    }
    else if (argc == 7) {
        parameters->n = 0;
        parameters->p = 0;
        for (int i = 1; i < argc; ) {
            if (strcmp(argv[i], "-r") == 0) {
                strcpy(parameters->edge_path, "./data/");
                strcat(parameters->edge_path,  argv[i + 1]);
                strcat(parameters->edge_path,  ".edges");
                i += 2;
                continue;
            }
            else if (strcmp(argv[i], "-k") == 0) {
                parameters->k = atoi(argv[i + 1]);
                i += 2;
                continue;
            }
            else if (strcmp(argv[i], "-o") == 0) {
                if (strcmp(argv[i + 1], "degeneracy") == 0)
                    parameters->order = DEGENERACY;
                else if (strcmp(argv[i + 1], "degree") == 0)
                    parameters->order = DEGREE;
                else if (strcmp(argv[i + 1], "random") == 0)
                    parameters->order = RANDOM;
                else if (strcmp(argv[i + 1], "learned") == 0) {
                    parameters->order = LEARNED;
                    strcpy(parameters->order_path, "./data/");
                    strcat(parameters->order_path,  argv[i + 1]);
                    strcat(parameters->order_path,  ".order");
                }
                else
                    parameters->order = LEXICOGRAPHIC;
                i += 2;
                continue;
            }
            else {
                fprintf(stderr, "Usage error.\n");
                exit(0);
            }
        }
    }
    else {
        fprintf(stderr, "Usage error.\n");
        exit(0);
    }
}
