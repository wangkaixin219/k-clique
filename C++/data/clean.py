file = input("file name:")
edge = set()
with open(file, "r") as f:
    for line in f.readlines():
        u, v = line.split()
        u = int(u)
        v = int(v)
        if u > v:
            u, v = v, u
        if (u, v) in edge:
            print(str(u) + " " + str(v) + " duplicates")
        elif u == v:
            print(str(u) + " self loop")
        else:
            edge.add((u, v))

with open("cleaned_"+file, "w") as f:
    for (u, v) in edge:
        f.write(str(u) + " " + str(v) + "\n")

