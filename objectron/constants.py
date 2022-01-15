
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
