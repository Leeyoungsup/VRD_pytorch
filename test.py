import SGRCG
import pandas as pd
import Detection
import time
start = time.time()
test_image_path = '../../data/dataset/test/image/'
test_caption_label = pd.read_csv(
    '../../data/dataset/test/caption_label.csv', encoding='cp949')
test_bbox_label = pd.read_csv(
    '../../data/dataset/test/bbox_label.csv', encoding='cp949')
whole_start_time = time.time()
Detection_info = Detection.Detection_test(
    test_image_path, test_bbox_label, test_caption_label)
SGRCG.SSG_test(Detection_info, test_image_path, test_caption_label)
print(f'total time: {time.time()-start}')
