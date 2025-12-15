import random
import shutil
import xml.etree.ElementTree as ET
from tqdm import tqdm
import os

sets = ['train', 'val']
# 这里使用要改成自己的类别
classes = ['airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge', 'chimney', 'dam',
           'Expressway-Service-area', 'Expressway-toll-station', 'golffield', 'groundtrackfield', 'harbor',
           'overpass', 'ship', 'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill']


def seg_img_type(xml_path, txt_path, source_images_path: str, train_val_percent=0.9, train_percent=0.7,
                 xml_deep: str = None):
    """
    XML文件按照一定的比例分成train、val、test三个数据集
    train_val_percent  表示的是train+val之和。若不需要test集则改为1
    train_percent 代表拿来训练的比例
    """
    source_jpg_path = os.path.join(xml_path, source_images_path)
    jpg_list = os.listdir(source_jpg_path)
    xml_path = os.path.join(xml_path, 'Annotations')
    if xml_deep is not None:
        xml_path = os.path.join(xml_path, xml_deep)
    txt_path = os.path.join(txt_path, 'seg')
    total_xml = os.listdir(xml_path)
    xml_num_list = [i.split('.')[0] for i in total_xml]
    jpg_num_list = [i.split('.')[0] for i in jpg_list]
    xml_exists_list = [i for i in xml_num_list if i in jpg_num_list]
    if not os.path.exists(txt_path):
        os.makedirs(txt_path)
    num = len(xml_exists_list)
    tv = int(num * train_val_percent)
    tr = int(tv * train_percent)
    trainval_list = random.sample(xml_exists_list, tv)
    train_list = random.sample(trainval_list, tr)
    val_list = [i for i in trainval_list if i not in train_list]
    test_list = [i for i in xml_exists_list if i not in trainval_list]

    file_trainval = open(txt_path + '/trainval.txt', 'w')
    file_test = open(txt_path + '/test.txt', 'w')
    file_train = open(txt_path + '/train.txt', 'w')
    file_val = open(txt_path + '/val.txt', 'w')

    for i in tqdm(trainval_list):
        file_trainval.write(i + '\n')
    for i in tqdm(train_list):
        file_train.write(i + '\n')
    for i in tqdm(val_list):
        file_val.write(i + '\n')
    for i in tqdm(test_list):
        file_test.write(i + '\n')
        
    file_trainval.close()
    file_train.close()
    file_val.close()
    file_test.close()


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    x = round(x, 6)
    w = round(w, 6)
    y = round(y, 6)
    h = round(h, 6)
    return x, y, w, h


def convert_annotation(xml_path, txt_path, image_id, is_difficult=True):
    in_file = open(os.path.join(xml_path, f'{image_id}.xml'), 'r', encoding='utf-8', )
    out_file = open(os.path.join(txt_path, f'{image_id}.txt'), 'w', encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        cls = obj.find('name').text
        if is_difficult:
            difficult = obj.find('difficult').text
            if cls not in classes or int(difficult) == 1:
                continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        b1, b2, b3, b4 = b
        # 标注越界修正
        if b2 > w:
            b2 = w
        if b4 > h:
            b4 = h
        b = (b1, b2, b3, b4)
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " +
                       " ".join([str(a) for a in bb]) + '\n')


def copy_img(source_jpg_path, img_path, image_id):
    shutil.copyfile(os.path.join(source_jpg_path, f'{image_id}.jpg'), os.path.join(img_path, f'{image_id}.jpg'))


def xml2txt(source_path, out_path, source_images_path, xml_deep: str = None, is_difficult=True):
    seg_path = os.path.join(out_path, 'seg')
    img_path = os.path.join(out_path, 'images')
    source_jpg_path = os.path.join(source_path, source_images_path)
    xml_path = os.path.join(source_path, 'Annotations')
    if xml_deep is not None:
        xml_path = os.path.join(xml_path, xml_deep)
    txt_path = os.path.join(out_path, 'label')
    if not os.path.exists(txt_path):
        os.makedirs(txt_path)
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    for image_set in sets:
        image_ids = open(os.path.join(seg_path, f'{image_set}.txt')).read().strip().split()
        list_file = open(os.path.join(out_path, f'{image_set}.txt'), 'w')
        for image_id in tqdm(image_ids):
            list_file.write(os.path.join(img_path, f'{image_id}.jpg\n'))
            convert_annotation(xml_path, txt_path, image_id, is_difficult)
            copy_img(source_jpg_path, img_path, image_id)
        list_file.close()


def split_dataset(source, output, source_images_path, xml_deep, is_difficult=True):
    seg_img_type(source, output, source_images_path, train_val_percent=1, xml_deep=xml_deep)
    xml2txt(source, output, source_images_path, xml_deep, is_difficult)


if __name__ == '__main__':
    source_dir = '/home/yjj/data/OpenDataLab___DIOR/raw/DIOR'
    output_dir = 'Datasets'
    split_dataset(source_dir, output_dir, "JPEGImages-trainval", xml_deep='Horizontal Bounding Boxes',
                  is_difficult=False)
