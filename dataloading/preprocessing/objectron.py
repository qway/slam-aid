from functools import partial
from pathlib import Path
import os
import requests
from tqdm import tqdm

base_url = "https://storage.googleapis.com/objectron"


def get_data_folder():
    current = Path.cwd()
    while not (current / "data").is_dir():
        current = current.parent
    return current / "data"


def download(url: str, fname: Path):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with fname.open('wb') as file, tqdm(
            desc=str(fname.parts[-3:]),
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
            leave=False
    ) as bar:
        for data in resp.iter_content(chunk_size=1024*4):
            size = file.write(data)
            bar.update(size)


def get_annotations(path: Path):
    annotations = {}
    for file in path.glob("*_*"):  # *_* to filter README.md
        with Path(file).open("r") as f:
            annotations[file.name] = [line.strip() for line in f.readlines()]
    return annotations


def download_and_save_batch_entry(batch_entry_id, data_path: Path):
    # Create folder for files
    [cls, folder] = batch_entry_id.split("/", maxsplit=1)
    folder = folder.replace("/", "-")
    save_path = data_path / folder
    save_path.mkdir(parents=True,
                    exist_ok=True)  # Note this works even if the path exists

    # Download Files
    video_filename = base_url + "/videos/" + batch_entry_id + "/video.MOV"
    metadata_filename = base_url + "/videos/" + batch_entry_id + "/geometry.pbdata"
    annotation_filename = base_url + "/annotations/" + batch_entry_id + ".pbdata"
    download(video_filename, save_path / "video.MOV")
    download(metadata_filename, save_path / "geometry.pbdata")
    download(annotation_filename, save_path / "annotation.pbdata")


if __name__ == '__main__':
    data_path = get_data_folder() / "objectron"
    annotations = get_annotations(
        get_data_folder().parent / "dataloading/preprocessing/objectron_index")
    all_data = []
    # Flatten for nicer progress bar
    for k, v in annotations.items():
        all_data += [(k, x) for x in v[:2]]

    for cls, entry in tqdm(all_data):
        download_and_save_batch_entry(entry, data_path / cls)
