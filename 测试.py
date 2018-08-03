#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
功能：
时间：
"""

import numpy as np


def ceshi():
    a = [9.775619208812713623e-02, -6.830719113349914551e-02]
    a2 = [9.775619208812713623e-02, -6.830719113349914551e-02]

    b = np.array(a)
    b2 = np.array(a2)

    b = np.row_stack((b, b2))
    np.savetxt("ceshi.txt", b, fmt="%.8f")

    x = np.loadtxt("ceshi.txt")
    print(x)
    np.savetxt("ceshi2.txt", x)


x = np.array([0, 1, 2])

for i in range(4):
    print(x[i])

if __name__ == "__main__":
    pass
