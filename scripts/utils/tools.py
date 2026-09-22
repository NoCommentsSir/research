def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            yield line.strip()