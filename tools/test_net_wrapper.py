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
    cat_file = ''

    with open(route) as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith('#'):
                continue

            if '=' in line:
                arg = line.split('=')
                if arg[0] == 'cat':
                    cat_file = arg[1]
                else:
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

    assert cat_file != '', 'Categories file not defined.'
    with open(cat_file) as f:
        inpt['categories'] = json.load(f)

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


def main(route):
    args, inpt, tmp_file_route = create_json(route)
    cmd_args = create_commandline_args(args, tmp_file_route)
    output_file_route = test_net.call_with_args(cmd_args)
    change_output(inpt, output_file_route)


if __name__ == '__main__':
    arg = check_args()
    main(arg)
