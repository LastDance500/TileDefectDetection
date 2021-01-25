# encoding:utf-8

# 导入依赖
import json
import os
import csv

# 读取 csv 文件
def read_csv(csv_path):
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        result = list(reader)
        return result

# 读取 json 文件
def read_json(anno_path):
    with open(anno_path) as f:
        load_list = json.load(f)
        return load_list

# 生成 json 文件
def save_json(full_dict, path):
    json.dump(full_dict, open(path, 'w'))

# 添加 images
def add_image(file_name, width, height, id):
    new_image = {}
    new_image['file_name'] = file_name
    new_image['width'] = width
    new_image['height'] = height
    new_image['id'] = id
    return new_image

# 添加 annotations
def add_annotations(segmentation, area, iscrowd, image_id, bbox, category_id, id):
    new_annotation = {}
    new_annotation['id'] = id
    new_annotation['image_id'] = image_id
    new_annotation['category_id'] = category_id
    new_annotation['segmentation'] = [segmentation]
    new_annotation['area'] = area
    new_annotation['bbox'] = bbox
    new_annotation['iscrowd'] = iscrowd
    return new_annotation

# 添加 categories
def add_category(index, label):
    category = {}
    category['supercategory'] = 'TianChi'
    category['id'] = index
    category['name'] = label
    return category

if __name__ == "__main__":
    # 文件地址
    csv_path = 'F:/train_imgs_crop_data(1).csv'
    anno_path = 'F:/BaiduNetdiskDownload/tile_round1_train_20201231/train_annos.json'
    train_path = 'F:/instances_train2014.json'
    val_path = 'F:/instances_val2014.json'

    # 各类数目
    count = {'1_1': 0, '1_2': 0, '1_3': 0, '1_4': 0, '1_5': 0, '1_6': 0, '2_1': 0, '2_2': 0, '2_3': 0, '2_4': 0, '2_5': 0, '2_6': 0, '3_1': 0, '3_2': 0, '3_3': 0, '3_4': 0, '3_5': 0, '3_6': 0}
    count_now = {'1_1': 0, '1_2': 0, '1_3': 0, '1_4': 0, '1_5': 0, '1_6': 0, '2_1': 0, '2_2': 0, '2_3': 0, '2_4': 0, '2_5': 0, '2_6': 0, '3_1': 0, '3_2': 0, '3_3': 0, '3_4': 0, '3_5': 0, '3_6': 0}

    # 准备两个字典, 用于判定有哪些图片已经加入
    train_set = {}
    val_set = {}

    # 两个字典, 一个存 train, 另一个存 val
    train_dict = {}
    val_dict = {}

    # 统计两个字典目前已存的图片个数
    train_count = 0
    val_count = 0

    # images annotations
    train_images = []
    train_annotations = []
    val_images = []
    val_annotations = []

    # 读取 json
    anno = read_json(anno_path)
    # print(anno)

    # 读取 csv
    csv = read_csv(csv_path)
    # print(csv)

    # 统计各类数目
    for i in range(len(anno)):
        category = anno[i].get('name').replace('.jpg', '')[-1] + "_" + str(anno[i].get('category'))
        count[category] += 1
        count_now[category] += 1
    print(count)

    # 对所有瑕疵进行处理
    n = len(anno)
    m = len(csv)

    for i in range(n):
        # 每隔 100 张瑕疵提示进度
        if i%100 == 0:
            print("Finish: {0}; To Do: {1}".format(i, n - i))

        # 图片名称
        name = anno[i].get('name')

        # 获得类别代码
        category = anno[i].get('name').replace('.jpg', '')[-1] + "_" + str(anno[i].get('category'))

        # 判断是否已经低于 1/3
        if count_now[category] > count[category]//3:
            # 数目更新
            count_now[category] -= 1

            # 找到对应的图片处理参数
            for j in range(m):
                # 对应的图片名
                if csv[j][1] == name:
                    # 判定图片是否已存入字典
                    if name not in train_set:
                        # 添加该图片
                        train_set[name] = train_count + 1

                        # 原始图片信息
                        org_height, org_width = anno[i].get('image_height'), anno[i].get('image_width')

                        # 原始边框位置
                        org_bbox = anno[i].get('bbox')

                        # 新的图片信息
                        height, width = tuple(eval(csv[j][4]))[:2]

                        # 新的图片起始位置
                        xmin, xmax, ymin, ymax = eval(csv[j][8]), eval(csv[j][9]), eval(csv[j][6]), eval(csv[j][7])

                        # 添加 images
                        train_images.append(add_image(file_name=name, width=width, height=height, id=train_set[name]))

                        # category_id
                        category_id = (int(name.replace(".jpg", "")[-1]) - 1) * 6 + anno[i].get('category')

                        # bbox
                        bbox = [org_bbox[0] - xmin, org_bbox[1] - ymin, org_bbox[2] - org_bbox[0], org_bbox[3] - org_bbox[1]]
                        
                        # segmentation
                        segmentation = [org_bbox[0] - xmin, org_bbox[1] - ymin, 
                                        org_bbox[2] - xmin, org_bbox[1] - ymin,
                                        org_bbox[0] - xmin, org_bbox[3] - ymin,
                                        org_bbox[2] - xmin, org_bbox[3] - ymin]
                        
                        # 添加 annotations
                        train_annotations.append(
                            add_annotations(
                                id=len(train_annotations) + 1,
                                image_id=train_set[name],
                                category_id=category_id, 
                                iscrowd=0, 
                                area=eval(csv[j][5]), 
                                bbox=bbox,
                                segmentation=segmentation)
                                )

                        # print(len(train_annotations) + 1)

                        # 更新计数
                        train_count += 1
                                
                    # 若已经在字典中
                    else:
                        # 原始图片信息
                        org_height, org_width = anno[i].get('image_height'), anno[i].get('image_width')

                        # 原始边框位置
                        org_bbox = anno[i].get('bbox')

                        # 新的图片信息
                        height, width = tuple(eval(csv[j][4]))[:2]

                        # 新的图片起始位置
                        xmin, xmax, ymin, ymax = eval(csv[j][8]), eval(csv[j][9]), eval(csv[j][6]), eval(csv[j][7])

                        # category_id
                        category_id = (int(name.replace(".jpg", "")[-1]) - 1) * 6 + anno[i].get('category')

                        # bbox
                        bbox = [org_bbox[0] - xmin, org_bbox[1] - ymin, org_bbox[2] - org_bbox[0], org_bbox[3] - org_bbox[1]]
                        
                        # segmentation
                        segmentation = [org_bbox[0] - xmin, org_bbox[1] - ymin, 
                                        org_bbox[2] - xmin, org_bbox[1] - ymin,
                                        org_bbox[0] - xmin, org_bbox[3] - ymin,
                                        org_bbox[2] - xmin, org_bbox[3] - ymin]
                        
                        # 添加 annotations
                        train_annotations.append(
                            add_annotations(
                                id=len(train_annotations) + 1,
                                image_id=train_set[name],
                                category_id=category_id, 
                                iscrowd=0, 
                                area=eval(csv[j][5]), 
                                bbox=bbox,
                                segmentation=segmentation)
                                )

                        # print(len(train_annotations) + 1)
        else:
            # 数目更新
            count_now[category] -= 1

            # 找到对应的图片处理参数
            for j in range(m):
                # 对应的图片名
                if csv[j][1] == name:
                    # 判定图片是否已存入字典
                    if name not in val_set:
                        # 添加该图片
                        val_set[name] = val_count + 1

                        # 原始图片信息
                        org_height, org_width = anno[i].get('image_height'), anno[i].get('image_width')

                        # 原始边框位置
                        org_bbox = anno[i].get('bbox')

                        # 新的图片信息
                        height, width = tuple(eval(csv[j][4]))[:2]

                        # 新的图片起始位置
                        xmin, xmax, ymin, ymax = eval(csv[j][8]), eval(csv[j][9]), eval(csv[j][6]), eval(csv[j][7])

                        # 添加 images
                        val_images.append(add_image(file_name=name, width=width, height=height, id=val_set[name]))

                        # category_id
                        category_id = (int(name.replace(".jpg", "")[-1]) - 1) * 6 + anno[i].get('category')

                        # bbox
                        bbox = [org_bbox[0] - xmin, org_bbox[1] - ymin, org_bbox[2] - org_bbox[0], org_bbox[3] - org_bbox[1]]
                        
                        # segmentation
                        segmentation = [org_bbox[0] - xmin, org_bbox[1] - ymin, 
                                        org_bbox[2] - xmin, org_bbox[1] - ymin,
                                        org_bbox[0] - xmin, org_bbox[3] - ymin,
                                        org_bbox[2] - xmin, org_bbox[3] - ymin]
                        
                        # 添加 annotations
                        val_annotations.append(
                            add_annotations(
                                id=len(val_annotations) + 1,
                                image_id=val_set[name],
                                category_id=category_id, 
                                iscrowd=0, 
                                area=eval(csv[j][5]), 
                                bbox=bbox,
                                segmentation=segmentation)
                                )

                        # print(len(val_annotations) + 1)

                        # 更新计数
                        val_count += 1
                                
                    # 若已经在字典中
                    else:
                        # 原始图片信息
                        org_height, org_width = anno[i].get('image_height'), anno[i].get('image_width')

                        # 原始边框位置
                        org_bbox = anno[i].get('bbox')

                        # 新的图片信息
                        height, width = tuple(eval(csv[j][4]))[:2]

                        # 新的图片起始位置
                        xmin, xmax, ymin, ymax = eval(csv[j][8]), eval(csv[j][9]), eval(csv[j][6]), eval(csv[j][7])

                        # category_id
                        category_id = (int(name.replace(".jpg", "")[-1]) - 1) * 6 + anno[i].get('category')

                        # bbox
                        bbox = [org_bbox[0] - xmin, org_bbox[1] - ymin, org_bbox[2] - org_bbox[0], org_bbox[3] - org_bbox[1]]
                        
                        # segmentation
                        segmentation = [org_bbox[0] - xmin, org_bbox[1] - ymin, 
                                        org_bbox[2] - xmin, org_bbox[1] - ymin,
                                        org_bbox[0] - xmin, org_bbox[3] - ymin,
                                        org_bbox[2] - xmin, org_bbox[3] - ymin]
                        
                        # 添加 annotations
                        val_annotations.append(
                            add_annotations(
                                id=len(val_annotations) + 1,
                                image_id=val_set[name],
                                category_id=category_id, 
                                iscrowd=0, 
                                area=eval(csv[j][5]), 
                                bbox=bbox,
                                segmentation=segmentation)
                                )

                        # print(len(val_annotations) + 1)
                
    # 添加 categories
    categories = []
    for i in range(1, 19):
        categories.append(add_category(i, str((i - 1)//6 + 1) + "_" + str(i - (i - 1)//6 * 6))) 

    # 添加到 train_dict, val_dict
    train_dict['annotations'] = train_annotations
    train_dict['images'] = train_images
    train_dict['categories'] = categories

    val_dict['annotations'] = val_annotations
    val_dict['images'] = val_images
    val_dict['categories'] = categories

    # 保存 json
    save_json(train_dict, train_path)
    save_json(val_dict, val_path)
