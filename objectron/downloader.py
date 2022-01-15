
import os
import re
import cv2
import timeit
import shutil
import random
import argparse
import requests
import functools
import multiprocessing

# local folder structure
DATA_DIR_PATH = 'objectron/data'
VIDEOS_TEMP_DIR_PATH = 'objectron/data/_videos'
CLASS_DIR_PATH = 'objectron/data/{class_}'
IMAGES_DIR_PATH = 'objectron/data/{class_}/images'
ANNOTATIONS_DIR_PATH = 'objectron/data/{class_}/annotations'
CLASSES = [
    'bike', 
    'book', 
    'bottle', 
    'camera', 
    'cereal_box', 
    'chair', 
    'cup', 
    'laptop', 
    'shoe',
]

# google cloud bucket urls
BASE_URL = 'https://storage.googleapis.com/objectron'
VIDEO_IDS_URL = 'https://storage.googleapis.com/objectron/v1/index/{class_}_annotations_{dataset}'
VIDEO_URL = 'https://storage.googleapis.com/objectron/videos/{video_id}/video.MOV'
METADATA_URL = 'https://storage.googleapis.com/objectron/videos/{video_id}/geometry.pbdata'
ANNOTATION_URL = 'https://storage.googleapis.com/objectron/annotations/{video_id}.pbdata'

def download(classes, videos_per_class, images_per_video, jobs):
    assert 1 <= classes <= len(CLASSES), f'there are maximum of {len(CLASSES)} classes'
    assert videos_per_class > 0, 'minimum of 1 video per class is required'
    assert images_per_video > 0, 'minimum of 1 image per video is required'
    assert jobs > 0, 'at least 1 job is required to start the process'

    # TODO: ask whether to continue
    if os.path.exists(DATA_DIR_PATH):
        print('removing previous data')
        shutil.rmtree(DATA_DIR_PATH)

    print('creating the folder structure')
    os.mkdir(DATA_DIR_PATH)
    os.mkdir(VIDEOS_TEMP_DIR_PATH)
    for class_ in CLASSES:
        os.mkdir(CLASS_DIR_PATH.format(class_=class_))
        os.mkdir(IMAGES_DIR_PATH.format(class_=class_))
        os.mkdir(ANNOTATIONS_DIR_PATH.format(class_=class_))

    classes_to_download = random.sample(CLASSES, classes)
    print(f'classes to download: {classes_to_download}')

    video_ids = []
    for class_ in classes_to_download:
        video_ids.extend(get_video_ids(class_, videos_per_class, 'train'))

    print(video_ids)

    start = timeit.default_timer()

    with multiprocessing.Pool(jobs) as pool:
        res = pool.map(
            functools.partial(extract_images, images_per_video=images_per_video),
            video_ids
        )
        print(res)

    end = timeit.default_timer()
    print(f'elapsed time: {end - start}')

    # TODO: remove videos directory

def extract_images(video_id, images_per_video):
    class_ = re.search(r'^([^/]+)/', video_id).group(1)

    response = requests.get(VIDEO_URL.format(video_id=video_id))
    video_path = os.path.join(VIDEOS_TEMP_DIR_PATH, f'{video_id_to_filename(video_id)}.MOV')
    with open(video_path, 'wb') as f:
        f.write(response.content)

    # TODO: extract image and annotation

    # metadata = requests.get(METADATA_URL.format(video_id=video_id))
    # annotation = requests.get(ANNOTATION_URL.format(video_id=video_id))

    # print(metadata)
    # print()
    # print(annotation)


def get_video_ids(class_, max_count, dataset):
    video_ids = requests.get(VIDEO_IDS_URL.format(class_=class_, dataset=dataset)).text
    video_ids = video_ids.split('\n')
    video_ids = random.sample(video_ids, min(max_count, len(video_ids)))

    return video_ids


def video_id_to_filename(video_id):
    return video_id.replace('/', '_')


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('--classes', type=int, default=9, help='number of classes to download')
    a.add_argument('--videos-per-class', type=int, default=10, help='how many videos per class to download')
    a.add_argument('--images-per-video', type=int, default=1, help='how many images to extract from each frame')
    a.add_argument('--jobs', type=int, default=1, help='how many threads to use for the processing')
    args = a.parse_args()
    
    download(args.classes, args.videos_per_class, args.images_per_video, args.jobs)
