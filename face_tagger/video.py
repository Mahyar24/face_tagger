import enum
import hashlib
import pathlib
import shutil
import subprocess
from typing import Optional


class FFmpegError(BaseException):
    """We Should terminate the program if this error is raised."""


class FrameMIME(enum.Enum):
    png = "png"
    jpg = "jpg"
    jpeg = "jpeg"


class Movie:
    def __init__(self, path: pathlib.Path) -> None:
        assert path.is_file(), f"{path!r} is not a valid file!"

        self.path = path
        self.abs_path = self.path.absolute()
        self.file_name = self.path.name
        self.mime = self.path.suffix
        self.hash = self.md5_hash()
        # Filling this attributes after `extract_frames` method call.
        self.frames_rate: Optional[int] = None
        self.frames_path: Optional[pathlib.Path] = None
        self.frames_mime: Optional[FrameMIME] = None

    def md5_hash(self, chunk_size: int = 2**24) -> str:
        md5 = hashlib.md5()
        with open(self.abs_path, "rb") as file:
            while True:
                data = file.read(chunk_size)
                if data:
                    md5.update(data)
                else:
                    break
        return md5.hexdigest()

    def extract_frames(
        self,
        directory: pathlib.Path,
        mime: FrameMIME = "jpeg",
        rate: int = 1,
        zero_pad: int = 6,
    ) -> pathlib.Path:
        assert shutil.which("ffmpeg") is not None, "Cannot find FFmpeg."

        rate = 1 / rate  # 5 -> 1 frame, every 5 sec.
        frames_dir = directory / f"frames_{self.hash}"

        if not (frames_dir.is_dir() and frames_dir.exists()):
            frames_dir.mkdir(parents=True)

        command = f"ffmpeg -i '{self.abs_path}' -r {rate} '{frames_dir.absolute()}/%0{zero_pad}d.{mime}'"
        return_code = subprocess.call(
            command, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL
        )

        if return_code:  # Error
            msg = "FFmpeg cannot extract frames!"
            if ":" in str(self.abs_path):
                msg += ' maybe because there is a ":" in filename!'
            raise FFmpegError(msg)

        self.frames_rate = rate
        self.frames_path = frames_dir.absolute()
        self.frames_mime = mime

        return frames_dir.absolute()
