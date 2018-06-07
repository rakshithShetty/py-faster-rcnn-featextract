#!/usr/bin/env python
import sys
import os
import json
import test_net

'''
#
def=/full-or-relative-path-to/models/coco/VGG16/faster_rcnn_end2end/test.prototxt
net=/full-or-relative-path-to/trainedModels/coco80Cls_vgg16_faster_rcnn_iter_290000.caffemodel
cfg=/full-or-relative-path-to/experiments/cfgs/faster_rcnn_end2end.yml

26 /full-or-relative-path-to/0:25.jpeg 672 384
70 /full-or-relative-path-to/0:69.jpeg 672 384
144 /full-or-relative-path-to/0:143.jpeg 672 384
270 /full-or-relative-path-to/0:269.jpeg 672 384
395 /full-or-relative-path-to/0:394.jpeg 672 384
664 /full-or-relative-path-to/0:663.jpeg 672 384
'''


def check_args():
    if len(sys.argv) == 1:
        print('Please specify an input file. I.e:')
        print('  ./test_net_wrapper.py input.txt')
        print('''
    This script need to be called as
      env LD_LIBRARY_PATH=$HOME/lib ./test_net_wrapper.py input.txt
    if it's running outside Triton
    ''')
        exit()

    return sys.argv[1]


def create_json(route):
    args = {}
    inpt = {'info': [],
            'images': [],
            'categories': []
            }
    tmp_path = '/tmp/'
    tmp_file_name = 'data.json'

    with open(route) as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith('#'):
                continue

            if '=' in line:
                arg = line.split('=')
                args[arg[0]] = arg[1]
            else:
                entry = {}
                s = line.split()
                entry['image_index'] = int(s[0])
                entry['image_id'] = s[1]
                entry['id'] = s[1].split('/')[-1]
                entry['width'] = int(s[2])
                entry['height'] = int(s[3])
                inpt['images'].append(entry)

    # category = {}
    # category['supercategory'] = 'blah'
    # category['id'] = 1
    # category['name'] = 'blah'

    inpt['categories'] = json.loads(
        '''
        [{"supercategory": "person",
     "id": 1,
     "name": "person"},
     {"supercategory": "vehicle",
     "id": 2,
     "name": "bicycle"},
     {"supercategory": "vehicle",
     "id": 3,
     "name": "car"},
     {"supercategory": "vehicle",
     "id": 4,
     "name": "motorcycle"},
     {"supercategory": "vehicle",
     "id": 5,
     "name": "airplane"},
     {"supercategory": "vehicle",
     "id": 6,
     "name": "bus"},
     {"supercategory": "vehicle",
     "id": 7,
     "name": "train"},
     {"supercategory": "vehicle",
     "id": 8,
     "name": "truck"},
     {"supercategory": "vehicle",
     "id": 9,
     "name": "boat"},
     {"supercategory": "outdoor",
     "id": 10,
     "name": "traffic light"},
     {"supercategory": "outdoor",
     "id": 11,
     "name": "fire hydrant"},
     {"supercategory": "outdoor",
     "id": 13,
     "name": "stop sign"},
     {"supercategory": "outdoor",
     "id": 14,
     "name": "parking meter"},
     {"supercategory": "outdoor",
     "id": 15,
     "name": "bench"},
     {"supercategory": "animal",
     "id": 16,
     "name": "bird"},
     {"supercategory": "animal",
     "id": 17,
     "name": "cat"},
     {"supercategory": "animal",
     "id": 18,
     "name": "dog"},
     {"supercategory": "animal",
     "id": 19,
     "name": "horse"},
     {"supercategory": "animal",
     "id": 20,
     "name": "sheep"},
     {"supercategory": "animal",
     "id": 21,
     "name": "cow"},
     {"supercategory": "animal",
     "id": 22,
     "name": "elephant"},
     {"supercategory": "animal",
     "id": 23,
     "name": "bear"},
    {"supercategory": "animal",
     "id": 24,
     "name": "zebra"},
     {"supercategory": "animal",
     "id": 25,
     "name": "giraffe"},
     {"supercategory": "accessory",
     "id": 27,
     "name": "backpack"},
     {"supercategory": "accessory",
     "id": 28,
     "name": "umbrella"},
     {"supercategory": "accessory",
     "id": 31,
     "name": "handbag"},
     {"supercategory": "accessory",
     "id": 32,
     "name": "tie"},
     {"supercategory": "accessory",
     "id": 33,
     "name": "suitcase"},
     {"supercategory": "sports",
     "id": 34,
     "name": "frisbee"},
     {"supercategory": "sports",
     "id": 35,
     "name": "skis"},
     {"supercategory": "sports",
     "id": 36,
     "name": "snowboard"},
     {"supercategory": "sports",
     "id": 37,
     "name": "sports ball"},
     {"supercategory": "sports",
     "id": 38,
     "name": "kite"},
     {"supercategory": "sports",
     "id": 39,
     "name": "baseball bat"},
     {"supercategory": "sports",
     "id": 40,
     "name": "baseball glove"},
     {"supercategory": "sports",
     "id": 41,
     "name": "skateboard"},
     {"supercategory": "sports",
     "id": 42,
     "name": "surfboard"},
     {"supercategory": "sports",
     "id": 43,
     "name": "tennis racket"},
     {"supercategory": "kitchen",
     "id": 44,
     "name": "bottle"},
     {"supercategory": "kitchen",
     "id": 46,
     "name": "wine glass"},
     {"supercategory": "kitchen",
     "id": 47,
     "name": "cup"},
     {"supercategory": "kitchen",
     "id": 48,
     "name": "fork"},
     {"supercategory": "kitchen",
     "id": 49,
     "name": "knife"},
    {"supercategory": "kitchen",
     "id": 50,
     "name": "spoon"},
     {"supercategory": "kitchen",
     "id": 51,
     "name": "bowl"},
     {"supercategory": "food",
     "id": 52,
     "name": "banana"},
     {"supercategory": "food",
     "id": 53,
     "name": "apple"},
     {"supercategory": "food",
     "id": 54,
     "name": "sandwich"},
     {"supercategory": "food",
     "id": 55,
     "name": "orange"},
     {"supercategory": "food",
     "id": 56,
     "name": "broccoli"},
     {"supercategory": "food",
     "id": 57,
     "name": "carrot"},
     {"supercategory": "food",
     "id": 58,
     "name": "hot dog"},
     {"supercategory": "food",
     "id": 59,
     "name": "pizza"},
     {"supercategory": "food",
     "id": 60,
     "name": "donut"},
     {"supercategory": "food",
     "id": 61,
     "name": "cake"},
     {"supercategory": "furniture",
     "id": 62,
     "name": "chair"},
     {"supercategory": "furniture",
     "id": 63,
     "name": "couch"},
     {"supercategory": "furniture",
     "id": 64,
     "name": "potted plant"},
     {"supercategory": "furniture",
     "id": 65,
     "name": "bed"},
     {"supercategory": "furniture",
     "id": 67,
     "name": "dining table"},
     {"supercategory": "furniture",
     "id": 70,
     "name": "toilet"},
     {"supercategory": "electronic",
     "id": 72,
     "name": "tv"},
     {"supercategory": "electronic",
     "id": 73,
     "name": "laptop"},
     {"supercategory": "electronic",
     "id": 74,
     "name": "mouse"},
     {"supercategory": "electronic",
     "id": 75,
     "name": "remote"},
     {"supercategory": "electronic",
     "id": 76,
     "name": "keyboard"},
     {"supercategory": "electronic",
     "id": 77,
     "name": "cell phone"},
     {"supercategory": "appliance",
     "id": 78,
     "name": "microwave"},
     {"supercategory": "appliance",
     "id": 79,
     "name": "oven"},
     {"supercategory": "appliance",
     "id": 80,
     "name": "toaster"},
     {"supercategory": "appliance",
     "id": 81,
     "name": "sink"},
     {"supercategory": "appliance",
     "id": 82,
     "name": "refrigerator"},
     {"supercategory": "indoor",
     "id": 84,
     "name": "book"},
     {"supercategory": "indoor",
     "id": 85,
     "name": "clock"},
     {"supercategory": "indoor",
     "id": 86,
     "name": "vase"},
     {"supercategory": "indoor",
     "id": 87,
     "name": "scissors"},
     {"supercategory": "indoor",
     "id": 88,
     "name": "teddy bear"},
     {"supercategory": "indoor",
     "id": 89,
     "name": "hair drier"},
     {"supercategory": "indoor",
     "id": 90,
     "name": "toothbrush"}]'''
    )

    with open(tmp_path + tmp_file_name, 'w') as f:
        json.dump(inpt, f)

    return args, inpt, tmp_path + tmp_file_name


def create_commandline_args(args, tmp_file_route):
    command_args = []
    for k, v in args.items():
        command_args.append('--' + k)
        command_args.append(v)
    command_args.append('--gpu')#put in file?
    command_args.append('0')#put in file?
    command_args.append('--imdb')#deprecated
    command_args.append('coco_2015_test-smb')#deprecated
    command_args.append('--json')
    command_args.append(tmp_file_route)
    command_args.append('--comp')#deprecated?
    command_args.append('--set')#deprecated
    command_args.append('EXP_DIR')#deprecated
    command_args.append('featExtract')#deprecated

    return command_args


def change_output(inpt, output_file_route):
    with open(output_file_route) as f:
        output = json.load(f)

    id_to_img = {i['id']: i for i in inpt['images']}

    for o in output:
        o['id'] = o['image_id']
        i = id_to_img.get(o['id'])

        if i is None:
            raise Exception('id ' + o['id'] + ' not found')

        o['image_index'] = i['image_index']
        o['image_id'] = i['image_id']
        o['height'] = i['height']
        o['width'] = i['width']

    with open(output_file_route, 'w') as f:
        json.dump(output, f)


# problem 1: we need to fetch the categories from somewhere (they come from imdb)

# dumpLocFeatures:
# problem 1: imdb is used in dumLocFeatures line 42 imdb._class_to_coco_cat_id.keys()
#               we need imdb._class_to_coco_cat_id and imdb._class_to_ind (they are different)
# problem 2: fix dump_SpatMapFeat if dumpClassOnly is not always 1
#   change imgid = ann['image_id'] in dumpLocFeatures, dump_clasDetFeat (in fact change all ann['blah'])
#   remove labels.txt from dumpLocFeatures
#   remove imdb from dumpLocFeatures


def main(route):
    args, inpt, tmp_file_route = create_json(route)
    cmd_args = create_commandline_args(args, tmp_file_route)
    output_file_route = test_net.call_with_args(cmd_args)
    change_output(inpt, output_file_route)


if __name__ == '__main__':
    arg = check_args()
    main(arg)
