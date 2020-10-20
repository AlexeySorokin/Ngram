def read_infile(infile):
    answer = []
    with open(infile, "r", encoding="utf8") as fin:
        for line in fin:
            line = line.strip()
            if line == "":
                continue
            splitted = line.split()
            if len(splitted) != 3:
                continue
            answer.append(splitted)
    return answer