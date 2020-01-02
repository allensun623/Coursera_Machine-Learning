# -*- coding: utf-8 -*-
import numpy as np

def test():
    #l = range(20)
    l = [[1, 3], [4, 5], [5, 7]]
    li =  np.ones((3, 2))
    lis =  np.ones(2)
    print(li)
    print(lis)
    print(li.shape)
    print(lis.shape)
    print(np.dot(l, lis))
    print(np.dot(l, lis.T))
    print(lis)
    print(lis.T)


def main():
    test()

if __name__ == "__main__":
    main()