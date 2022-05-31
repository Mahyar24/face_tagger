import pathlib
from concurrent.futures import ALL_COMPLETED, ProcessPoolExecutor, wait

import face_recognition as fr
import numpy as np


def make_encoding(person_dir: pathlib.Path) -> list[np.ndarray]:
    assert person_dir.is_dir(), f"cannot find '{person_dir}'"

    faces = []
    for img in person_dir.iterdir():
        if img.is_file():
            faces.append(fr.face_encodings(fr.load_image_file(img.absolute()))[0])
    return faces


def make_encodings(
    path: pathlib.Path, semaphore_num: int = 16
) -> dict[str, list[np.ndarray]]:
    assert path.is_dir(), f"cannot find '{path}'"

    tasks = {}

    with ProcessPoolExecutor(max_workers=semaphore_num) as executor:
        for person_dir in path.iterdir():
            if person_dir.is_dir():
                tasks[person_dir.name] = executor.submit(make_encoding, person_dir)
        wait(tasks.values(), return_when=ALL_COMPLETED)

    return {k: v.result() for k, v in tasks.items()}
