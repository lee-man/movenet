import os.path
from os import listdir
from PIL import Image

def pinjie( ):
    imgs = [Image.open(os.path.join(p,fn)) for fn in listdir(p) if fn.endswith('.jpg')]
    ims = []
    '''for j in imgs:
        im = Image.fromarray(j)

        nimgs.append(im)'''
    for i in imgs:
        #new_img = i.resize((1280, 1280), Image.BILINEAR)
        ims.append(i)
    # 单幅图像尺寸
    width, height = ims[0].size
    # 创建空白长图
    result = Image.new(ims[0].mode, (width, int(height * (len(ims)/2.2))))
    # 拼接图片
    result.paste(ims[0], box=(0, 0))
    result.paste(ims[1],box=(0,height))
    result.paste(ims[2], box=(ims[1].size[0]+1, height))

    # 保存图片
    result.save('/Users/rachel/PycharmProjects/movenet/experiments/1127test/res2.jpg')


p = '/Users/rachel/PycharmProjects/movenet/experiments/1127test/n_im/'
pinjie()