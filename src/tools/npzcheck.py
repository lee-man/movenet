import numpy as np
import os
'''p2 = '/Users/rachel/PycharmProjects/movenet/experiments/1222test/npy'
filelist = []
def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            yield f
for i in findAllFile(p2):
    filelist.append(i)
content = []
for j in filelist:
    a = np.load(os.path.join(p2,j))
    b = a.reshape(17,64,64)
    content.append(b)
np.save('/Users/rachel/PycharmProjects/movenet/experiments/1222test/total.npy',content)
'''
a = np.load('/Users/rachel/PycharmProjects/movenet/experiments/1222test/total.npy')
print(1)

