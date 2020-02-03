import numpy as np

my_predictions = np.array([11, 2, 5, 3, 6, 65, 7])

# sort predictions descending
pre=np.array([[idx, p] for idx, p in enumerate(my_predictions)])
post = pre[pre[:,1].argsort()[::-1]]
sort_array = [len(my_predictions)-i-1 for i in range(len(my_predictions))]
post2 = pre[sort_array]
r = post[:,1]
ix = post[:,0]
print(pre)
print(pre[:,1].argsort()[::-1])
print(post)
print(post2)
my_predictions[1] = 0
print(post)

print(r)
print(ix)

