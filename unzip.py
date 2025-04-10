import os
import struct
import cv2 as cv
import numpy as np
import matplotlib.pyplot as pl

data_dir = "dataset"

train = 'train'
test = 'test'

train_image_file = 'train-images-idx3-ubyte'
train_label_file = 'train-labels-idx1-ubyte'
test_image_file = 't10k-images-idx3-ubyte'
test_label_file = 't10k-labels-idx1-ubyte'

def decompress_label(label_file_path):
    with open(label_file_path, 'rb') as f:
        magic_number = int.from_bytes(f.read(4), byteorder='big')
        image_number = int.from_bytes(f.read(4), byteorder='big')
        fmt = '>B'
        byte_num = 1
        label_list = []
        for i in range(image_number):
            label_list.append(struct.unpack(fmt, f.read(byte_num))[0])
        # print(number_list)
        return label_list


def decompress_image(image_file_path):
    with open(image_file_path, 'rb') as f:
        magic_number = int.from_bytes(f.read(4), byteorder='big')
        image_number = int.from_bytes(f.read(4), byteorder='big')
        height = int.from_bytes(f.read(4), byteorder='big')
        width = int.from_bytes(f.read(4), byteorder='big')
        print('number: %d height: %d  width: %d' % (image_number, height, width))
        img_len = height * width
        fmt = '>' + str(img_len) + 'B'

        image_list = []
        for i in range(image_number):
            img = struct.unpack(fmt, f.read(img_len))
            img = np.reshape(img, (height, width))
            image_list.append(img)
        return image_list


def decompress_image_save(image_file_path, label_list, res_dir):
    with open(image_file_path, 'rb') as f:
        magic_number = int.from_bytes(f.read(4), byteorder='big')
        image_number = int.from_bytes(f.read(4), byteorder='big')
        height = int.from_bytes(f.read(4), byteorder='big')
        width = int.from_bytes(f.read(4), byteorder='big')
        print('number: %d height: %d  width: %d' % (image_number, height, width))
        img_len = height * width
        fmt = '>' + str(img_len) + 'B'

        nums = [0 for i in range(10)]
        for i in range(image_number):
            img = struct.unpack(fmt, f.read(img_len))
            img = np.reshape(img, (height, width))

            image_dir = os.path.join(res_dir, str(label_list[i]))
            if not os.path.exists(image_dir):
                os.mkdir(image_dir)
            image_path = os.path.join(image_dir, str(nums[label_list[i]]) + ".png")
            cv.imwrite(image_path, img)
            nums[label_list[i]] += 1


def decompress():
    des_dir = data_dir
    des_train_dir = os.path.join(des_dir, train)
    des_test_dir = os.path.join(des_dir, test)
    print('mkdir result dir ok')

    # label_list = decompress_label(os.path.join(data_dir, train_label_file))
    # decompress_image_save(os.path.join(data_dir, train_image_file), label_list, des_train_dir)
    print('train image and label ok')

    label_list = decompress_label(os.path.join(data_dir, test_label_file))
    decompress_image_save(os.path.join(data_dir, test_image_file), label_list, des_test_dir)
    print('test image and label ok')



if __name__ == '__main__':
    decompress()