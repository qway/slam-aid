import glob
import os
import subprocess

# import box as Box
from pathlib import Path

import decord
import numpy as np
import torch

from torch.utils.data import IterableDataset
import matplotlib.pyplot as plt
import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# The annotations are stored in protocol buffer format.
from objectron.schema import annotation_data_pb2 as annotation_protocol
# The AR Metadata captured with each frame in the video
from objectron.dataset import graphics


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


def grab_frame(video_file, frame_ids):
    """Grab an image frame from the video file."""
    reader = decord.VideoReader(video_file)
    frames = reader.get_batch(frame_ids)
    return frames


class ObjectronDataset(IterableDataset):
    def __init__(self, folders, batchsize=4, resolution=(1440, 1920)):
        # Get all files
        videos = []
        self.annotations = []
        metadatas = []
        for folder in folders:
            videos.append(str(folder / "video.MOV"))
            with (folder / "annotation.pbdata").open('rb') as pb:
                sequence = annotation_protocol.Sequence()
                sequence.ParseFromString(pb.read())
                self.annotations.append(sequence)
            metadatas.append(folder / "geometry.pbdata")

        self.vl = decord.VideoLoader(videos,
                                     ctx=decord.cpu(),
                                     shape=(batchsize, resolution[0], resolution[1], 3),
                                     interval=20,
                                     shuffle=1,
                                     skip=50,
                                     )

    def __len__(self):
        return len(self.vl)

    def __iter__(self):
        def iterator():
            for batch in self.vl:
                ids = batch[1][:,0]
                frames = batch[1][:,1]

                keypoints3d = []
                keypoints2d = []
                labels = []
                for id, frame in zip(ids, frames):
                    (object_keypoints_2d, object_keypoints_3d, object_categories, keypoint_size_list,
                     annotation_types) = get_frame_annotation(self.annotations[id], frame)
                    keypoints2d.append(object_keypoints_2d)
                    keypoints3d.append(object_keypoints_3d)
                    labels.append(annotation_types)

                yield batch[0], torch.Tensor(keypoints2d), torch.Tensor(keypoints3d), labels
        return iterator()

def get_leaf_dirs(dirname):
    folders = []
    def fast_scandir(dirname):
        subfolders = [f.path for f in os.scandir(dirname) if f.is_dir()]
        if subfolders:
            for dirname in subfolders:
                fast_scandir(dirname)
        else:
            folders.append(Path(dirname))
        return subfolders
    fast_scandir(dirname)
    return folders

if __name__ == '__main__':
    decord.bridge.set_bridge("torch")  # Gives native pytorch arrays from decord

    """annotation_file = '/home/bene/projects/slam-aid/data/objectron/bike_annotations/batch-0-0/annotation.pbdata'
    video_filename = '/home/bene/projects/slam-aid/data/objectron/bike_annotations/batch-0-0/video.MOV'
    geometry_filename = '/home/bene/projects/slam-aid/data/objectron/bike_annotations/batch-0-0/geometry.pbdata'  # a.k.a. AR metadata
    frame_id = 100
    with open(annotation_file, 'rb') as pb:
        sequence = annotation_protocol.Sequence()
        sequence.ParseFromString(pb.read())
        frame = grab_frame(video_filename, [frame_id])
        annotation, cat, num_keypoints, types = get_frame_annotation(sequence, frame_id)
        image = graphics.draw_annotation_on_image(frame[0].numpy(), annotation,
                                                  num_keypoints)
        imgplot = plt.imshow(image)
    plt.show()"""

    paths = get_leaf_dirs("/home/bene/projects/slam-aid/data/objectron")[:6]
    batchsize=4
    dset = ObjectronDataset(paths, batchsize=batchsize)
    print(len(dset))
    for i, batch in enumerate(dset):
        img, projbb2d, bb3d, label = batch
        figure = plt.figure(figsize=(8, 8))
        for i in range(batchsize):
            plt.title(label[i])
            figure.add_subplot(1, batchsize, i+1)
            plt.axis("off")
            image = graphics.draw_annotation_on_image(img[i].numpy(), projbb2d[i].numpy(),
                                                      [9])
            plt.imshow(image)
        plt.show()
        #print(f"{img} : {bb3d}")
