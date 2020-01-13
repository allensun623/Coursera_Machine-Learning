# -*- coding: utf-8 -*-
import numpy as np

def test():
    #l = range(20)
    l = [[1, 3], [4, 5], [5, 7]]
    lis =  np.asarray([(0, 1, 2), (3, 4, 5), (6, 7, 8)])
    x = lis[1,]
    y = lis[:2]
    z = lis[:, 1]
    print(lis)
    print(x)
    print(y)
    print(z)

def main():
    test()

if __name__ == "__main__":
    main()