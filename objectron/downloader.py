
import os
import re
import cv2
import tqdm
import pickle
import struct
import timeit
import shutil
import random
import argparse
import requests
import functools
import multiprocessing
import numpy as np

from objectron.schema import object_pb2 as object_protocol
from objectron.schema import annotation_data_pb2 as annotation_protocol
from objectron.schema import a_r_capture_metadata_pb2 as ar_metadata_protocol
import objectron.constants as constants

def download(classes, videos_per_class, images_per_video, jobs):
    assert 1 <= classes <= len(constants.CLASSES), f'there is a total of {len(constants.CLASSES)} classes'
    assert videos_per_class > 0, 'minimum of 1 video per class is required'
    assert images_per_video > 0, 'minimum of 1 image per video is required'
    assert jobs > 0 or jobs == -1, 'at least 1 job is required to start the process'

    # TODO: ask whether to continue
    if os.path.exists(constants.DATA_DIR_PATH):
        print('removing previous data')
        shutil.rmtree(constants.DATA_DIR_PATH)

    print('creating folder structure')
    os.mkdir(constants.DATA_DIR_PATH)
    os.mkdir(constants.VIDEOS_TEMP_DIR_PATH)
    for class_ in constants.CLASSES:
        os.mkdir(constants.CLASS_DIR_PATH.format(class_=class_))
        os.mkdir(constants.IMAGES_DIR_PATH.format(class_=class_))
        os.mkdir(constants.ANNOTATIONS_DIR_PATH.format(class_=class_))

    classes_to_download = random.sample(constants.CLASSES, classes)
    print(f'classes to download: {", ".join(classes_to_download)}')

    video_ids = []
    for class_ in classes_to_download:
        video_ids.extend(get_video_ids(class_, videos_per_class, 'train'))

    start = timeit.default_timer()

    if jobs == -1:
        jobs = multiprocessing.cpu_count()

    with multiprocessing.Pool(processes=jobs) as pool:
        with tqdm.tqdm(total=len(video_ids)) as pbar:
            fun = functools.partial(extract_images, images_per_video=images_per_video)
            for i, _ in enumerate(pool.imap_unordered(fun, video_ids)):
                pbar.update()

    end = timeit.default_timer()
    print(f'time elapsed: {end - start} s')

    # TODO: remove videos directory

def extract_images(video_id, images_per_video):
    class_ = re.search(r'^([^/]+)/', video_id).group(1)

    response = requests.get(constants.ANNOTATION_URL.format(video_id=video_id))
    annotation_sequence = annotation_protocol.Sequence()
    annotation_sequence.ParseFromString(response.content)

    response = requests.get(constants.METADATA_URL.format(video_id=video_id))
    geometry_data = get_geometry_data(response.content)

    response = requests.get(constants.VIDEO_URL.format(video_id=video_id))
    video_path = os.path.join(constants.VIDEOS_TEMP_DIR_PATH, f'{video_id_to_filename(video_id)}.MOV')
    with open(video_path, 'wb') as f:
        f.write(response.content)

    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = list(range(total_frames))
    frame_indices = random.sample(frame_indices, min(images_per_video, total_frames))
    frame_indices.sort()

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        frame_filename = f'{video_id_to_filename(video_id)}_frame-{idx}'
        image_path = os.path.join(constants.IMAGES_DIR_PATH.format(class_=class_), f'{frame_filename}.jpg')
        cv2.imwrite(image_path, frame)

        object_keypoints_2d, object_keypoints_3d, object_categories, \
            keypoint_size_list, annotation_types = get_frame_annotation(annotation_sequence, idx)

        transform, projection, view = geometry_data[idx]

        annotation_dict = dict(
            object_keypoints_2d=object_keypoints_2d,
            object_keypoints_3d=object_keypoints_3d,
            object_categories=object_categories,
            keypoint_size_list=keypoint_size_list,
            annotation_types=annotation_types,
            transform=transform,
            projection=projection,
            view=view
        )

        annotation_path = os.path.join(constants.ANNOTATIONS_DIR_PATH.format(class_=class_), f'{frame_filename}.pickle')
        with open(annotation_path, 'wb') as f:
            pickle.dump(annotation_dict, f)

    os.remove(video_path)


def get_frame_annotation(sequence, frame_id):
  """Grab an annotated frame from the sequence."""
  data = sequence.frame_annotations[frame_id]
  object_id = 0
  object_keypoints_2d = []
  object_keypoints_3d = []
  object_rotations = []
  object_translations = []
  object_scale = []
  num_keypoints_per_object = []
  object_categories = []
  annotation_types = []
  # Get the camera for the current frame. We will use the camera to bring
  # the object from the world coordinate to the current camera coordinate.
  camera = np.array(data.camera.transform).reshape(4, 4)

  for obj in sequence.objects:
    rotation = np.array(obj.rotation).reshape(3, 3)
    translation = np.array(obj.translation)
    object_scale.append(np.array(obj.scale))
    transformation = np.identity(4)
    transformation[:3, :3] = rotation
    transformation[:3, 3] = translation
    obj_cam = np.matmul(camera, transformation)
    object_translations.append(obj_cam[:3, 3])
    object_rotations.append(obj_cam[:3, :3])
    object_categories.append(obj.category)
    annotation_types.append(obj.type)

  keypoint_size_list = []
  for annotations in data.annotations:
    num_keypoints = len(annotations.keypoints)
    keypoint_size_list.append(num_keypoints)
    for keypoint_id in range(num_keypoints):
      keypoint = annotations.keypoints[keypoint_id]
      object_keypoints_2d.append(
          (keypoint.point_2d.x, keypoint.point_2d.y, keypoint.point_2d.depth))
      object_keypoints_3d.append(
          (keypoint.point_3d.x, keypoint.point_3d.y, keypoint.point_3d.z))
    num_keypoints_per_object.append(num_keypoints)
    object_id += 1

  return (object_keypoints_2d, object_keypoints_3d, object_categories, keypoint_size_list,
          annotation_types)


def get_geometry_data(bytes):
    sequence_geometry = []

    i = 0
    frame_number = 0

    while i < len(bytes):
        # Read the first four Bytes in little endian '<' integers 'I' format
        # indicating the length of the current message.
        msg_len = struct.unpack('<I', bytes[i:i + 4])[0]
        i += 4
        message_buf = bytes[i:i + msg_len]
        i += msg_len
        frame_data = ar_metadata_protocol.ARFrame()
        frame_data.ParseFromString(message_buf)


        transform = np.reshape(frame_data.camera.transform, (4, 4))
        projection = np.reshape(frame_data.camera.projection_matrix , (4, 4))
        view = np.reshape(frame_data.camera.view_matrix , (4, 4))

        sequence_geometry.append((transform, projection, view))

    return sequence_geometry


def get_video_ids(class_, max_count, dataset):
    video_ids = requests.get(constants.VIDEO_IDS_URL.format(class_=class_, dataset=dataset)).text
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
    a.add_argument('--jobs', type=int, default=-1, help='how many threads to use for the processing; -1 for max available')
    args = a.parse_args()
    
    download(args.classes, args.videos_per_class, args.images_per_video, args.jobs)
