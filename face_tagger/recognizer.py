import pathlib
import pickle
import re
import statistics
from concurrent.futures import ALL_COMPLETED, ProcessPoolExecutor, wait
from typing import Literal, Union

import face_recognition as fr
import numpy as np


def make_encoding(
    person_dir: pathlib.Path,
    model: Union[Literal["cnn"], Literal["hog"]] = "hog",
    num_jitters: int = 100,
) -> list[np.ndarray]:
    assert person_dir.is_dir(), f"cannot find '{person_dir}'"

    faces = []
    for img in person_dir.iterdir():
        if img.is_file():
            faces.append(
                fr.face_encodings(
                    fr.load_image_file(img.absolute()),
                    num_jitters=num_jitters,
                    model=model,
                )[0]
            )
    return faces


def make_encodings(
    path: pathlib.Path,
    model: Union[Literal["cnn"], Literal["hog"]] = "hog",
    num_jitters: int = 100,
    semaphore_num: int = 16,
) -> dict[str, list[np.ndarray]]:
    assert path.is_dir(), f"cannot find '{path}'"

    tasks = {}

    with ProcessPoolExecutor(max_workers=semaphore_num) as executor:
        for person_dir in path.iterdir():
            if person_dir.is_dir():
                tasks[person_dir.name] = executor.submit(
                    make_encoding, person_dir, model, num_jitters
                )
        wait(tasks.values(), return_when=ALL_COMPLETED)

    return {k: v.result() for k, v in tasks.items()}


def recognize_image(
    image: pathlib.Path,
    encodings: dict[str, list[np.ndarray]],
    model: Union[Literal["cnn"], Literal["hog"]] = "hog",
    loc_model: Union[Literal["cnn"], Literal["hog"]] = "hog",
    hmean_tolerance: float = 0.6,
    accept_tolerance: float = 0.5,
    num_jitters: int = 1,
) -> list[bool]:
    img = fr.load_image_file(image)
    img_locations = fr.face_locations(img, model=loc_model)
    img_encodings = fr.face_encodings(
        img, img_locations, num_jitters=num_jitters, model=model
    )

    results = np.zeros((len(encodings),), dtype=bool)
    for img_encoding in img_encodings:
        guesses = {}
        for i, encoding in enumerate(encodings.values()):
            distances = fr.face_distance(encoding, img_encoding)
            if (
                min_distance := min(distances)
            ) <= accept_tolerance or statistics.harmonic_mean(
                distances
            ) <= hmean_tolerance:
                guesses[i] = min_distance

        if guesses:
            results[min(guesses, key=guesses.get)] = True

    return results


def recognize_images(
    images_dir: pathlib.Path,
    encodings: dict[str, list[np.ndarray]],
    model: Union[Literal["cnn"], Literal["hog"]] = "hog",
    loc_model: Union[Literal["cnn"], Literal["hog"]] = "hog",
    hmean_tolerance: float = 0.6,
    accept_tolerance: float = 0.5,
    num_jitters: int = 1,
    semaphore_num: int = 8,
) -> np.ndarray:
    assert images_dir.is_dir(), f"cannot find '{images_dir}'"

    tasks = {}

    with ProcessPoolExecutor(max_workers=semaphore_num) as executor:
        for img in images_dir.iterdir():
            if img.is_file():
                tasks[img.name] = executor.submit(
                    recognize_image,
                    img,
                    encodings,
                    model,
                    loc_model,
                    hmean_tolerance,
                    accept_tolerance,
                    num_jitters,
                )
        wait(tasks.values(), return_when=ALL_COMPLETED)

    return np.vstack([tasks[k].result() for k in sorted(tasks.keys())])


def show_names(
    array: np.ndarray,
    encodings: dict[str, list[np.ndarray]],
) -> None:
    for line in array:
        result = " ".join(np.where(line, list(encodings.keys()), "")).strip()
        result = re.sub(r"\s\s+", ", ", result)
        if not result:
            result = None
        print("->", result)


if __name__ == "__main__":
    # known_encodings = make_encodings(pathlib.Path("/home/mahyar/test/Pics/"))
    # with open('/home/mahyar/test/PICS.PICKLE', 'wb') as file:
    #     pickle.dump(known_encodings, file, protocol=pickle.HIGHEST_PROTOCOL)

    with open("/home/mahyar/test/PICS.PICKLE", "rb") as file:
        known_encodings = pickle.load(file)

    import time
    s = time.perf_counter()
    x = recognize_images(pathlib.Path("/home/mahyar/test/2/f"), known_encodings)
    # show_names(x, known_encodings)
    print(time.perf_counter() - s)

    with open("/home/mahyar/test/arr.np", "wb") as file:
        np.save(file, x)
