./tools/test_net.py --gpu 0 --def models/coco/VGG16/faster_rcnn_end2end/test.prototxt --net trainedModels/coco80Cls_vgg16_faster_rcnn_iter_290000.caffemodel --imdb coco_2015_test --cfg experiments/cfgs/faster_rcnn_end2end.yml --comp --set EXP_DIR featExtract