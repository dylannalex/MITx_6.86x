with open("stopwords.txt") as f:
    lines = f.readlines()
    lines = [line.replace("\n", "") for line in lines]

print(lines)
