import numpy as np

with open('movie_ids.txt', 'r+', encoding = "ISO-8859-1") as file:
    data = file.readlines()
data = [item.rstrip('\n').split(' ', 1)[1] for item in data]
print(data)
print(np.size(data))