import os

from PIL import Image

from util import algorithm


def image_to_txt(image_dir, out_path, has_tag=True, img_num=0):
    if has_tag:
        with open(out_path, 'w') as f:
            for parent, dirnames, filenames in os.walk(image_dir):
                for file in filenames:
                    image_path = os.path.join(parent, file)
                    if image_path.endswith('png'):
                        img = Image.open(image_path)
                        for x in range(28):
                            for y in range(28):
                                f.write(str(img.getpixel((x, y))/100) + ' ')
                            f.write(str(ord(parent[-1]) - ord('A')))
                        f.write('\n')
    else:
        with open(out_path, 'w') as f:
            for i in range(1, int(img_num) + 1):
                img_path = r'{0}/{1}.png'.format(image_dir, i)
                img = Image.open(img_path)
                for x in range(28):
                    for y in range(28):
                        f.write(str(img.getpixel((x,y))/100) + ' ')
                f.write('\n')

def load_data(path, model):
    x = []
    y = []
    with open(path) as f:
        for l in f:
            arr = list(map(lambda _x: float(_x), l.split()))
            x.append(arr[:-1])
            if model == 'image':
                y.append(algorithm.int_to_list(int(arr[-1]), 8))
            elif model == 'sin':
                y.append(arr[-1:])
            elif model == 'LCD':
                y.append(algorithm.int_to_list(int(arr[-1]), ))
    return x, y


def load_input(path):
    x = []
    with open(path) as f:
        for l in f:
            arr = list(map(lambda _x: float(_x), l.split()))
            x.append(arr)

    return x
