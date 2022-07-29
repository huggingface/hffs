import os
import tempfile
from functools import partial
from pathlib import PurePosixPath
from typing import Optional

import fsspec
from fsspec.utils import stringify_path
from huggingface_hub import HfFolder, delete_file, hf_hub_url, upload_file
from huggingface_hub.constants import REPO_TYPES
from huggingface_hub.hf_api import repo_info


class HfFileSystem(fsspec.AbstractFileSystem):
    """
    Access a remote Hugging Face Hub repository as if were a local file system.

    Args:
        repo_id (`str`):
            The remote repository to access as if were a local file system,
            for example: `"username/custom_transformers"`
        token (`str`, *optional*):
            Authentication token, obtained with `HfApi.login` method. Will
            default to the stored token.
        repo_type (`str`, *optional*):
            Set to `"dataset"` or `"space"` if the remote repositry is a dataset or
            space repositroy, `None` or `"model"` if it is a model repository. Default is
            `None`.
        revision (`str`, *optional*):
            An optional Git revision id which can be a branch name, a tag, or a
            commit hash. Defaults to the head of the `"main"` branch.

    Direct usage:

    ```python
    >>> import hffs

    >>> fs = hffs.HfFileSystem("username/my-dataset", repo_type="dataset")

    >>> # Read a remote file
    >>> with fs.open("remote/file/in/repo.bin") as f:
    ...     data = f.read()

    >>> # Write a remote file
    >>> with fs.open("remote/file/in/repo.bin", "wb") as f:
    ...     f.write(data)
    ```

    Usage via [`fsspec`](https://filesystem-spec.readthedocs.io/en/latest/)):

    ```python
    >>> import fsspec

    >>> # Read a remote file
    >>> with fsspec.open("hf://username/my-dataset:/remote/file/in/repo.bin", repo_type="dataset") as f:
    ...     data = f.read()

    >>> # Write a remote file
    >>> with fsspec.open("hf://username/my-dataset:/remote/file/in/repo.bin", "wb", repo_type="dataset") as f:
    ...     f.write(data)
    ```
    """

    root_marker = ""
    protocol = "hf"

    def __init__(
        self,
        repo_id: str,
        token: Optional[str] = None,
        repo_type: Optional[str] = None,
        revision: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(self, **kwargs)

        if repo_type not in REPO_TYPES:
            raise ValueError(f"Invalid repo type, must be one of {REPO_TYPES}")

        self.repo_id = repo_id
        self.token = token if token is not None else HfFolder.get_token()
        self.repo_type = repo_type
        self.revision = revision

    def _dircache_from_repo_info(self):
        repo_info_obj = repo_info(self.repo_id, revision=self.revision, repo_type=self.repo_type, token=self.token)
        for sibling in repo_info_obj.siblings:
            child = {
                "name": sibling.rfilename,
                "size": None,  # waiting for #951
                "type": "file",
            }
            for parent in list(PurePosixPath(sibling.rfilename).parents)[:-1] + [self.root_marker]:
                self.dircache.setdefault(str(parent), []).append(child)
                child = {"name": str(parent), "size": None, "type": "directory"}

    def invalidate_cache(self, path=None):
        self.dircache.clear()

    @classmethod
    def _strip_protocol(cls, path):
        if isinstance(path, list):
            return [cls._strip_protocol(stringify_path(p)) for p in path]
        if path.startswith(f"{cls.protocol}://"):
            path = path[len(f"{cls.protocol}://") :]
            if ":/" in path:
                _, path = path.split(":/", 1)
                path = path.lstrip("/")
            else:
                path = cls.root_marker
        return path

    def unstrip_protocol(self, path):
        return super().unstrip_protocol(
            f"{self.repo_id}{'@' + self.revision if self.revision is not None else ''}:/{path}"
        )

    @staticmethod
    def _get_kwargs_from_urls(path):
        protocol = HfFileSystem.protocol
        if path.startswith(f"{protocol}://"):
            path = path[len(f"{protocol}://") :]
        out = {}
        if ":/" in path:
            out["repo_id"], _ = path.split(":/", 1)
        else:
            out["repo_id"] = path
        if "@" in out["repo_id"]:
            out["repo_id"], out["revision"] = out["repo_id"].split("@", 1)
        return out

    def _open(
        self,
        path: str,
        mode: str = "rb",
        **kwargs,
    ):
        if mode == "rb":
            url = hf_hub_url(
                self.repo_id,
                path,
                repo_type=self.repo_type,
                revision=self.revision,
            )
            return fsspec.open(
                url,
                mode=mode,
                headers={"authorization": f"Bearer {self.token}"},
            ).open()
        else:
            return TempFileUploader(self, path, mode=mode)

    def _rm(self, path):
        path = self._strip_protocol(path)
        delete_file(
            path_in_repo=path,
            repo_id=self.repo_id,
            token=self.token,
            repo_type=self.repo_type,
            revision=self.revision,
            commit_message=f"Delete {path} with hffs",
        )
        self.invalidate_cache()

    def rm(self, path, recursive=False, maxdepth=None):
        paths = self.expand_path(path, recursive=recursive, maxdepth=maxdepth)
        paths = [path for path in paths if not self.isdir(path)]
        for path in paths:
            self.rm_file(path)

    def ls(self, path, detail=False, **kwargs):
        path = self._strip_protocol(path)
        if not self.dircache:
            self._dircache_from_repo_info()
        out = self._ls_from_cache(path)
        if out is None:
            raise FileNotFoundError(path)
        if detail:
            return out
        return [o["name"] for o in out]

    def cp_file(self, path1, path2, **kwargs):
        path1 = self._strip_protocol(path1)
        path2 = self._strip_protocol(path2)

        with self.open(path1, "rb") as f1:
            with self.open(path2, "wb") as f2:
                for block in iter(partial(f1.read, self.blocksize), b""):
                    f2.write(block)


class TempFileUploader(fsspec.spec.AbstractBufferedFile):
    def _initiate_upload(self):
        self.temp_file = tempfile.NamedTemporaryFile(delete=False)
        if "a" in self.mode:
            with self.fs.open(self.path, "rb") as f:
                for block in iter(partial(f.read, self.blocksize), b""):
                    self.temp_file.write(block)

    def _upload_chunk(self, final=False):
        self.buffer.seek(0)
        block = self.buffer.read()
        self.temp_file.write(block)
        if final:
            self.temp_file.close()
            upload_file(
                path_or_fileobj=self.temp_file.name,
                path_in_repo=self.path,
                repo_id=self.fs.repo_id,
                token=self.fs.token,
                repo_type=self.fs.repo_type,
                revision=self.fs.revision,
                commit_message=f"Upload {self.path} with hffs",
            )
            os.remove(self.temp_file.name)
            self.fs.invalidate_cache()
