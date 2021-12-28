import os
filePath = '/Users/rachel/PycharmProjects/movenet/experiments/1203test/worse'
x = os.listdir(filePath)
p = 'yyliu@linux9.cse.cuhk.edu.hk:/research/d4/rshr/yyliu/code/movenet/data/crop_square/images/'
y = []
print('scp -r ',end='')
for i in x:
    j = p+i+' \\'
    print(j)
print('/Users/rachel/PycharmProjects/movenet/experiments/1205test/1202worse')