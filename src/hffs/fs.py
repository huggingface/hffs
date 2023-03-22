import itertools
import os
import tempfile
from typing import Dict, Optional, Tuple
from urllib.parse import quote

import fsspec
import huggingface_hub
import huggingface_hub.constants
import huggingface_hub.utils
import huggingface_hub.utils._pagination
import requests

from . import __version__


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
        endpoint (`str`, *optional*):
            The endpoint to use. If not provided, the default one (https://huggingface.co) is used.
        token (`str`, *optional*):
            Authentication token, obtained with `HfApi.login` method. Will
            default to the stored token.
        revision (`str`, *optional*):
            An optional Git revision id which can be a branch name, a tag, or a
            commit hash. Defaults to the head of the `"main"` branch.

    Usage:

    ```python
    >>> import hffs

    >>> fs = hffs.HfFileSystem()

    >>> # List files
    >>> fs.glob("my-username/my-model/*.bin")
    ["pytorch_model.bin"]
    >>> fs.ls("datasets/my-username/my-dataset", detail=False)
    ['.gitattributes', 'README.md', 'data.json']

    >>> # Read/write files
    >>> with fs.open("my-username/my-model/pytorch_model.bin") as f:
    ...     data = f.read()
    >>> with fs.open("my-username/my-model/pytorch_model.bin", "wb") as f:
    ...     f.write(data)
    ```
    """

    root_marker = ""
    protocol = "hf"

    def __init__(
        self,
        *args,
        endpoint: Optional[str] = None,
        token: Optional[str] = None,
        revision: Optional[str] = None,
        **storage_options,
    ):
        super().__init__(*args, **storage_options)
        self.endpoint = endpoint or huggingface_hub.constants.ENDPOINT
        self.token = token
        self.revision = revision
        self._api = huggingface_hub.HfApi(
            endpoint=endpoint, token=token, library_name="hffs", library_version=__version__
        )
        self._repository_type_and_id_exists_cache: Dict[Tuple[str, str], bool] = {}

    def _repo_exists(self, repo_id: str, repo_type: str) -> bool:
        if (repo_type, repo_id) in self._repository_type_and_id_exists_cache:
            return self._repository_type_and_id_exists_cache[(repo_type, repo_id)]
        else:
            try:
                self._api.repo_info(repo_id, repo_type=repo_type)
                self._repository_type_and_id_exists_cache[(repo_type, repo_id)] = True
                return True
            except (huggingface_hub.utils.RepositoryNotFoundError, huggingface_hub.utils.HFValidationError):
                self._repository_type_and_id_exists_cache[(repo_type, repo_id)] = False
                return False

    def _resolve_repo_id(self, path: str) -> Tuple[str, str, str]:
        path = self._strip_protocol(path)
        if not path:
            # can't list repositories at root
            raise NotImplementedError("Acces to repositories lists is not implemented")
        elif path.split("/")[0] + "/" in huggingface_hub.constants.REPO_TYPES_URL_PREFIXES.values():
            if "/" not in path:
                # can't list repositories at the repository type level
                raise NotImplementedError("Acces to repositories lists is not implemented.")
            repo_type, path = path.split("/", 1)
            repo_type = huggingface_hub.constants.REPO_TYPES_MAPPING[repo_type]
        else:
            repo_type = huggingface_hub.constants.REPO_TYPE_MODEL
        if path.count("/") > 0:
            repo_id_with_namespace = "/".join(path.split("/")[:2])
            path_in_repo_with_namespace = "/".join(path.split("/")[2:])
            repo_id_without_namespace = path.split("/")[0]
            path_in_repo_without_namespace = "/".join(path.split("/")[1:])
            if self._repo_exists(repo_id_with_namespace, repo_type=repo_type):
                repo_id = repo_id_with_namespace
                path_in_repo = path_in_repo_with_namespace
            elif self._repo_exists(repo_id_without_namespace, repo_type=repo_type):
                repo_id = repo_id_without_namespace
                path_in_repo = path_in_repo_without_namespace
            else:
                raise FileNotFoundError(f"No such repository: '{repo_id_with_namespace}'")
        else:
            if self._repo_exists(path, repo_type=repo_type):
                repo_id = path
                path_in_repo = ""
            else:
                # can't list repositories at the namespace level
                raise NotImplementedError("Acces to repositories lists is not implemented.")

        return repo_type, repo_id, path_in_repo

    def invalidate_cache(self, path=None):
        if not path:
            self.dircache.clear()
            self._repository_type_and_id_exists_cache.clear()
        else:
            path = self._strip_protocol(path)
            while path:
                self.dircache.pop(path, None)
                path = self._parent(path)

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
        repo_type, repo_id, path_in_repo = self._resolve_repo_id(path)
        path = self._strip_protocol(path)
        operations = [huggingface_hub.CommitOperationDelete(path_in_repo=path_in_repo)]
        commit_message = f"Delete {path}"
        self._api.create_commit(
            repo_id=repo_id,
            repo_type=repo_type,
            token=self.token,
            operations=operations,
            revision=self.revision,
            commit_message=kwargs.get("commit_message", commit_message),
            commit_description=kwargs.get("commit_description"),
        )
        self.invalidate_cache(path=path)

    def rm(self, path, recursive=False, maxdepth=None, **kwargs):
        repo_type, repo_id, _ = self._resolve_repo_id(path)
        root_path = huggingface_hub.constants.REPO_TYPES_URL_PREFIXES.get(repo_type, "") + repo_id
        paths = self.expand_path(path, recursive=recursive, maxdepth=maxdepth)
        paths_in_repo = [path[len(root_path) + 1 :] for path in paths if not self.isdir(path)]
        operations = [
            huggingface_hub.CommitOperationDelete(path_in_repo=path_in_repo) for path_in_repo in paths_in_repo
        ]
        commit_message = f"Delete {path} "
        commit_message += "recursively " if recursive else ""
        commit_message += f"up to depth {maxdepth} " if maxdepth is not None else ""
        # TODO: use `commit_description` to list all the deleted paths?
        self._api.create_commit(
            repo_id=repo_id,
            repo_type=repo_type,
            token=self.token,
            operations=operations,
            revision=self.revision,
            commit_message=kwargs.get("commit_message", commit_message),
            commit_description=kwargs.get("commit_description"),
        )
        self.invalidate_cache(path=path)

    def ls(self, path, detail=True, refresh=False, **kwargs):
        path = self._strip_protocol(path)
        if path not in self.dircache or refresh:
            repo_type, repo_id, path_in_repo = self._resolve_repo_id(path)
            path_prefix = huggingface_hub.constants.REPO_TYPES_URL_PREFIXES.get(repo_type, "") + repo_id + "/"
            tree_iter = self._iter_tree(path)
            try:
                tree_item = next(tree_iter)
            except huggingface_hub.utils.EntryNotFoundError:
                if "/" in path_in_repo:
                    path = self._parent(path)
                    tree_iter = self._iter_tree(path)
                else:
                    raise
            else:
                tree_iter = itertools.chain([tree_item], tree_iter)
            child_infos = []
            for tree_item in tree_iter:
                child_info = {
                    "name": path_prefix + tree_item["path"],
                    "size": tree_item["size"],
                    "type": tree_item["type"],
                }
                if tree_item["type"] == "file":
                    child_info.update(
                        {
                            "blob_id": tree_item["oid"],
                            "lfs": tree_item.get("lfs"),
                            "last_modified": huggingface_hub.utils.parse_datetime(tree_item["lastCommit"]["date"]),
                        },
                    )
                child_infos.append(child_info)
            self.dircache[path] = child_infos
        out = self._ls_from_cache(path)
        return out if detail else [o["name"] for o in out]

    def _iter_tree(self, path: str):
        path = self._strip_protocol(path)
        repo_type, repo_id, path_in_repo = self._resolve_repo_id(path)
        revision = self.revision if self.revision is not None else huggingface_hub.constants.DEFAULT_REVISION
        path = (
            f"{self._api.endpoint}/api/{repo_type}s/{repo_id}/tree/{quote(revision, safe='')}/{path_in_repo}"
        ).rstrip("/")
        headers = self._api._build_hf_headers()
        yield from huggingface_hub.utils._pagination.paginate(path, params=None, headers=headers)

    def cp_file(self, path1, path2, **kwargs):
        repo_type1, repo_id1, path_in_repo1 = self._resolve_repo_id(path1)
        path1 = self._strip_protocol(path1)
        repo_type2, repo_id2, path_in_repo2 = self._resolve_repo_id(path2)
        path2 = self._strip_protocol(path2)

        same_repo = repo_type1 == repo_type2 and repo_id1 == repo_id2

        # TODO: Wait for https://github.com/huggingface/huggingface_hub/issues/1083 to be resolved to simplify this logic
        if same_repo and self.info(path1)["lfs"] is not None:
            headers = self._api._build_hf_headers(is_write_action=True)
            commit_message = f"Copy {path1} to {path2}"
            payload = {
                "summary": kwargs.get("commit_message", commit_message),
                "description": kwargs.get("commit_description", ""),
                "files": [],
                "lfsFiles": [
                    {
                        "path": path_in_repo2,
                        "algo": "sha256",
                        "oid": self.info(path1)["lfs"]["oid"],
                    }
                ],
                "deletedFiles": [],
            }
            revision = self.revision if self.revision is not None else huggingface_hub.constants.DEFAULT_REVISION
            r = requests.post(
                f"{self.endpoint}/api/{repo_type1}s/{repo_id1}/commit/{quote(revision, safe='')}",
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
                path_in_repo=path_in_repo2,
                repo_id=repo_id2,
                token=self.token,
                repo_type=repo_type2,
                revision=self.revision,
                commit_message=kwargs.get("commit_message", commit_message),
                commit_description=kwargs.get("commit_description"),
            )
        self.invalidate_cache(path1)
        self.invalidate_cache(path2)

    def modified(self, path, refresh=False):
        info = self.info(path, refresh=refresh)
        if info["type"] != "file":
            raise FileNotFoundError(path)
        return info["last_modified"]

    def info(self, path, **kwargs):
        path = self._strip_protocol(path)
        _, _, path_in_repo = self._resolve_repo_id(path)
        if not path_in_repo:
            return {"name": path, "size": None, "type": "directory"}
        return super().info(path, **kwargs)


class HfFile(fsspec.spec.AbstractBufferedFile):
    def __init__(self, fs: HfFileSystem, path: str, **kwargs):
        super().__init__(fs, path, **kwargs)
        self.fs: HfFileSystem
        self.repo_type, self.repo_id, self.path_in_repo = fs._resolve_repo_id(path)

    def _fetch_range(self, start, end):
        headers = {
            "range": f"bytes={start}-{end - 1}",
            **self.fs._api._build_hf_headers(),
        }
        url = _path_to_http_url(
            self.path_in_repo,
            self.repo_id,
            endpoint=self.fs.endpoint,
            repo_type=self.repo_type,
            revision=self.fs.revision,
        )
        r = huggingface_hub.utils.http_backoff("GET", url, headers=headers)
        huggingface_hub.utils.hf_raise_for_status(r)
        return r.content

    def _initiate_upload(self):
        self.temp_file = tempfile.NamedTemporaryFile(prefix="hffs-", delete=False)

    def _upload_chunk(self, final=False):
        self.buffer.seek(0)
        block = self.buffer.read()
        self.temp_file.write(block)
        if final:
            self.temp_file.close()
            commit_message = f"Upload {self.path}"
            self.fs._api.upload_file(
                path_or_fileobj=self.temp_file.name,
                path_in_repo=self.path_in_repo,
                repo_id=self.repo_id,
                token=self.fs.token,
                repo_type=self.repo_type,
                revision=self.fs.revision,
                commit_message=self.kwargs.get("commit_message", commit_message),
                commit_description=self.kwargs.get("commit_description"),
            )
            os.remove(self.temp_file.name)
            self.fs.invalidate_cache(path=self.path)
