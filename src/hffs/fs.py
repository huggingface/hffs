import collections
import os
import tempfile
from pathlib import PurePosixPath
from typing import Dict, Optional, Tuple
from urllib.parse import quote

import fsspec
import huggingface_hub
import huggingface_hub.constants
import huggingface_hub.utils
import requests


def _repo_type_and_id_from_hf_id(hf_id: str) -> Tuple[Optional[str], Optional[str], str]:
    """
    Returns the repo type and ID from a HF ID
    Args:
        hf_id (`str`):
            An URL or ID of a repository on the HF hub. Accepted values are:
            - <repo_type>/<namespace>/<repo_id>
            - <namespace>/<repo_id>
            - <repo_id>
    Returns:
        A tuple with two items: repo_type (`str` or `None`), namespace (`str` or
        `None`) and repo_id (`str`).
    """
    url_segments = hf_id.split("/")
    is_hf_id = len(url_segments) <= 3
    if is_hf_id:
        if len(url_segments) == 3:
            # Passed <repo_type>/<user>/<model_id> or <repo_type>/<org>/<model_id>
            repo_type, namespace, repo_id = url_segments[-3:]
        elif len(url_segments) == 2:
            # Passed <user>/<model_id> or <org>/<model_id>
            namespace, repo_id = hf_id.split("/")[-2:]
            repo_type = None
        else:
            # Passed <model_id>
            repo_id = url_segments[0]
            namespace, repo_type = None, None
    else:
        raise ValueError(f"Unable to retrieve user and repo ID from the passed HF ID: {hf_id}")

    if repo_type not in huggingface_hub.constants.REPO_TYPES:
        assert repo_type is not None, "repo_type `None` do not have mapping"
        repo_type = huggingface_hub.constants.REPO_TYPES_MAPPING.get(repo_type)

    return repo_type, namespace, repo_id


# huggingface_hub.hf_hub_url doesn't support non-default endpoints at the moment
# tracked in https://github.com/huggingface/huggingface_hub/issues/1082
def _path_to_http_url(
    path_in_repo: str,
    repo_id: str,
    endpoint: Optional[str] = None,
    repo_type: Optional[str] = None,
    revision: Optional[str] = None,
) -> str:
    if repo_type not in huggingface_hub.constants.REPO_TYPES:
        raise ValueError(f"Invalid repo type, must be one of {huggingface_hub.constants.REPO_TYPES}")

    if repo_type in huggingface_hub.constants.REPO_TYPES_URL_PREFIXES:
        repo_id = huggingface_hub.constants.REPO_TYPES_URL_PREFIXES[repo_type] + repo_id

    if endpoint is None:
        endpoint = huggingface_hub.constants.ENDPOINT

    if revision is None:
        revision = huggingface_hub.constants.DEFAULT_REVISION

    return f"{endpoint}/{repo_id}/resolve/{quote(revision, safe='')}/{quote(path_in_repo, safe='')}"


class HfFileSystem(fsspec.AbstractFileSystem):
    """
    Access a remote Hugging Face Hub repository as if were a local file system.

    Args:
        repo_id (`str`):
            The remote repository to access as if were a local file system,
            for example: `"username/custom_transformers"`
        endpoint (`str`, *optional*):
            The endpoint to use. If not provided, the default one (https://huggingface.co) is used.
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
    >>> with fsspec.open("hf://dataset/my-username/my-dataset:/remote/file/in/repo.bin") as f:
    ...     data = f.read()

    >>> # Write a remote file
    >>> with fsspec.open("hf://dataset/my-username/my-dataset:/remote/file/in/repo.bin", "wb") as f:
    ...     f.write(data)
    ```
    """

    root_marker = ""
    protocol = "hf"

    @huggingface_hub.utils.validate_hf_hub_args
    def __init__(
        self,
        repo_id: str,
        token: Optional[str] = None,
        repo_type: Optional[str] = None,
        revision: Optional[str] = None,
        endpoint: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(self, **kwargs)

        if repo_type not in huggingface_hub.constants.REPO_TYPES:
            raise ValueError(f"Invalid repo type, must be one of {huggingface_hub.constants.REPO_TYPES}")

        self.repo_id = repo_id
        self.endpoint = endpoint if endpoint is not None else huggingface_hub.constants.ENDPOINT
        self.token = token if token is not None else huggingface_hub.HfFolder.get_token()
        self.repo_type = repo_type if repo_type is not None else huggingface_hub.constants.REPO_TYPE_MODEL
        self.revision = revision
        self._api = huggingface_hub.HfApi(endpoint=endpoint, token=token)

    def _dircache_from_repo_info(self):
        repo_info = self._api.repo_info(
            self.repo_id, revision=self.revision, repo_type=self.repo_type, token=self.token, files_metadata=True
        )
        child_dirs = collections.defaultdict(set)
        for repo_file in repo_info.siblings:
            child = {
                "name": repo_file.rfilename,
                "size": repo_file.size,
                "type": "file",
                # extra metadata for files
                "blob_id": repo_file.blob_id,
                "lfs": repo_file.lfs,
            }
            parents = list(PurePosixPath(repo_file.rfilename).parents)[:-1] + [self.root_marker]
            parent = str(parents[0])
            self.dircache.setdefault(str(parent), []).append(child)
            child = {"name": parent, "size": None, "type": "directory"}
            for parent in parents[1:]:
                parent = str(parent)
                if child["name"] in child_dirs[parent]:
                    break
                self.dircache.setdefault(parent, []).append(child)
                child_dirs[parent].add(child["name"])
                child = {"name": parent, "size": None, "type": "directory"}
        return self.dircache

    def invalidate_cache(self, path=None):
        # TODO: use `path` to optimize cache invalidation -> requires filtering on the server to be implemented efficiently
        self.dircache.clear()

    @classmethod
    def _strip_protocol(cls, path):
        if isinstance(path, list):
            return [cls._strip_protocol(fsspec.utils.stringify_path(p)) for p in path]
        protocol = cls.protocol
        if path.startswith(protocol + "://"):
            path = path[len(protocol + "://") :]
            hf_id, *paths = path.split(":/", 1)
            path = paths[0] if paths else cls.root_marker
        return path

    def unstrip_protocol(self, path):
        return super().unstrip_protocol(
            f"{self.repo_type}/{self.repo_id}{'@' + self.revision if self.revision is not None else ''}:/{path}"
        )

    @staticmethod
    def _get_kwargs_from_urls(path):
        protocol = HfFileSystem.protocol
        if path.startswith(protocol + "://"):
            path = path[len(protocol + "://") :]
        hf_id, *paths = path.split(":/", 1)
        hf_id, *revisions = hf_id.split("@", 1)
        out = {}
        repo_type, namespace, repo_id = _repo_type_and_id_from_hf_id(hf_id)
        repo_id = f"{namespace}/{repo_id}" if namespace is not None else repo_id
        out["repo_id"] = repo_id
        if repo_type is not None:
            out["repo_type"] = repo_type
        if revisions:
            out["revision"] = revisions[0]
        return out

    def _open(
        self,
        path: str,
        mode: str = "rb",
        **kwargs,
    ):
        if mode == "ab":
            raise NotImplementedError("Appending to remote files is not yet supported.")
        path = self._strip_protocol(path)
        return HfFile(self, path, mode=mode, **kwargs)

    def _rm(self, path, **kwargs):
        path = self._strip_protocol(path)
        operations = [huggingface_hub.CommitOperationDelete(path_in_repo=path)]
        commit_message = f"Delete {path}"
        self._api.create_commit(
            repo_id=self.repo_id,
            repo_type=self.repo_type,
            token=self.token,
            operations=operations,
            revision=self.revision,
            commit_message=kwargs.pop("commit_message", commit_message),
            commit_description=kwargs.pop("commit_description", None),
        )
        self.invalidate_cache()

    def rm(self, path, recursive=False, maxdepth=None, **kwargs):
        paths = self.expand_path(path, recursive=recursive, maxdepth=maxdepth)
        file_paths = [path for path in paths if not self.isdir(path)]
        operations = [huggingface_hub.CommitOperationDelete(path_in_repo=file_path) for file_path in file_paths]
        commit_message = f"Delete {path} "
        commit_message += "recursively " if recursive else ""
        commit_message += f"up to depth {maxdepth} " if maxdepth is not None else ""
        # TODO: use `commit_description` to list all the deleted paths?
        self._api.create_commit(
            repo_id=self.repo_id,
            repo_type=self.repo_type,
            token=self.token,
            operations=operations,
            revision=self.revision,
            commit_message=kwargs.pop("commit_message", commit_message),
            commit_description=kwargs.pop("commit_description", None),
        )
        self.invalidate_cache()

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

        # TODO: Wait for https://github.com/huggingface/huggingface_hub/issues/1083 to be resolved to simplify this logic
        if self.info(path1)["lfs"] is not None:
            headers = huggingface_hub.utils.build_hf_headers(token=self.token, is_write_action=True)
            commit_message = f"Copy {path1} to {path2}"
            payload = {
                "summary": kwargs.pop("commit_message", commit_message),
                "description": kwargs.pop("commit_description", ""),
                "files": [],
                "lfsFiles": [
                    {
                        "path": path2,
                        "algo": "sha256",
                        "oid": self.info(path1)["lfs"]["sha256"],
                    }
                ],
                "deletedFiles": [],
            }
            revision = self.revision if self.revision is not None else huggingface_hub.constants.DEFAULT_REVISION
            r = requests.post(
                f"{self.endpoint}/api/{self.repo_type}s/{self.repo_id}/commit/{quote(revision, safe='')}",
                json=payload,
                headers=headers,
            )
            huggingface_hub.utils.hf_raise_for_status(r)
        else:
            with self.open(path1, "rb") as f:
                content = f.read()
            commit_message = f"Copy {path1} to {path2}"
            self._api.upload_file(
                path_or_fileobj=content,
                path_in_repo=path2,
                repo_id=self.repo_id,
                token=self.token,
                repo_type=self.repo_type,
                revision=self.revision,
                commit_message=kwargs.pop("commit_message", commit_message),
                commit_description=kwargs.pop("commit_description", None),
            )
        self.invalidate_cache()


class HfFile(fsspec.spec.AbstractBufferedFile):
    def _fetch_range(self, start, end):
        headers = {
            "range": f"bytes={start}-{end}",
            **huggingface_hub.utils.build_hf_headers(token=self.fs.token),
        }
        url = _path_to_http_url(
            self.path,
            self.fs.repo_id,
            endpoint=self.fs.endpoint,
            repo_type=self.fs.repo_type,
            revision=self.fs.revision,
        )
        r = huggingface_hub.utils.http_backoff("GET", url, headers=headers)
        huggingface_hub.utils.hf_raise_for_status(r)
        return r.content

    def _initiate_upload(self):
        self.temp_file = tempfile.NamedTemporaryFile(dir=huggingface_hub.constants.HUGGINGFACE_HUB_CACHE, delete=False)

    def _upload_chunk(self, final=False):
        self.buffer.seek(0)
        block = self.buffer.read()
        self.temp_file.write(block)
        if final:
            self.temp_file.close()
            commit_message = f"Upload {self.path}"
            self.fs._api.upload_file(
                path_or_fileobj=self.temp_file.name,
                path_in_repo=self.path,
                repo_id=self.fs.repo_id,
                token=self.fs.token,
                repo_type=self.fs.repo_type,
                revision=self.fs.revision,
                commit_message=self.kwargs.pop("commit_message", commit_message),
                commit_description=self.kwargs.pop("commit_description", None),
            )
            os.remove(self.temp_file.name)
            self.fs.invalidate_cache()

