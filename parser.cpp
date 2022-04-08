
#include "defs.h"

void arg_parser(int argc, const char* argv[], param& parameters) {
    if (argc == 4) {
        if (string(argv[1]) == "-g") {
            parameters.n = stoul(argv[2]);
            parameters.p = stod(argv[3]);
            parameters.path = "./data/syn.edges";
        }
        else {
            cout << "Usage error." << endl;
            exit(0);
        }
    }
    else if (argc == 7) {
        parameters.n = 0;
        parameters.p = 0;
        for (int i = 1; i < argc; ) {
            if (string(argv[i]) == "-r") {
                parameters.path = "./data/" + string(argv[i + 1]);
                i += 2;
                continue;
            }
            else if (string(argv[i]) == "-k") {
                parameters.k = stoul(argv[i + 1]);
                i += 2;
                continue;
            }
            else if (string(argv[i]) == "-o") {
                if (string(argv[i + 1]) == "degeneracy")
                    parameters.order = DEGENERACY;
                else if (string(argv[i + 1]) == "degree")
                    parameters.order = DEGREE;
                else if (string(argv[i + 1]) == "random")
                    parameters.order = RANDOM;
                else if (string(argv[i + 1]) == "learned")
                    parameters.order = LEARNED;
                else
                    parameters.order = LEXICOGRAPHIC;
                i += 2;
                continue;
            }
            else {
                cout << "Usage error." << endl;
                exit(0);
            }
        }
    }
    else {
        cout << "Usage error." << endl;
        exit(0);
    }
}
