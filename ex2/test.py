# -*- coding: utf-8 -*-
import numpy as np

def test():
    #l = range(20)
    l = [[1, 2], [3, 4]]
    ll = np.asarray(l)
    print('type of ll', type(ll))
    print(ll)
    out = map_feature(ll[0,:], ll[1,:])
    print(out)    


def map_feature(x1, x2, degree=6):
    X_array = np.array([x1**(i-j) * x2**j for i in range(1,degree+1) for j in range(i+1)])
    X = np.concatenate((np.ones((np.size(x1), 1)), X_array.T), axis=1)
    return X

def main():
    test()

if __name__ == "__main__":
    main()