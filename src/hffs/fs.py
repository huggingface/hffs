import itertools
import os
import tempfile
from glob import has_magic
from typing import Dict, Optional, Tuple
from urllib.parse import quote, unquote

import fsspec
import huggingface_hub
import huggingface_hub.constants
import huggingface_hub.utils
import huggingface_hub.utils._pagination
import requests

from . import __version__


class HfFileSystem(fsspec.AbstractFileSystem):
    """
    Access a remote Hugging Face Hub repository as if were a local file system.

    Args:
        endpoint (`str`, *optional*):
            The endpoint to use. If not provided, the default one (https://huggingface.co) is used.
        token (`str`, *optional*):
            Authentication token, obtained with `HfApi.login` method. Will default to the stored token.

    Usage:

    ```python
    >>> import hffs

    >>> fs = hffs.HfFileSystem()

    >>> # List files
    >>> fs.glob("my-username/my-model/*.bin")
    ['my-username/my-model/pytorch_model.bin']
    >>> fs.ls("datasets/my-username/my-dataset", detail=False)
    ['datasets/my-username/my-dataset/.gitattributes', 'datasets/my-username/my-dataset/README.md', 'datasets/my-username/my-dataset/data.json']

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
        **storage_options,
    ):
        super().__init__(*args, **storage_options)
        self.endpoint = endpoint or huggingface_hub.constants.ENDPOINT
        self.token = token
        self._api = huggingface_hub.HfApi(
            endpoint=endpoint, token=token, library_name="hffs", library_version=__version__
        )
        self._repo_and_revision_exists_cache: Dict[Tuple[str, str, str], Tuple[bool, bool]] = {}

    def _repo_and_revision_exists(self, repo_id: str, repo_type: str, revision: str) -> Tuple[bool, bool]:
        revision = revision if revision != huggingface_hub.constants.DEFAULT_REVISION else None
        if (repo_type, repo_id, revision) in self._repo_and_revision_exists_cache:
            return self._repo_and_revision_exists_cache[(repo_type, repo_id, revision)]
        else:
            try:
                self._api.repo_info(repo_id, repo_type=repo_type, revision=revision)
            except (huggingface_hub.utils.RepositoryNotFoundError, huggingface_hub.utils.HFValidationError):
                self._repo_and_revision_exists_cache[(repo_type, repo_id, revision)] = False, False
            except huggingface_hub.utils.RevisionNotFoundError:
                self._repo_and_revision_exists_cache[(repo_type, repo_id, revision)] = True, False
                self._repo_and_revision_exists_cache[(repo_type, repo_id, None)] = True, True
            else:
                self._repo_and_revision_exists_cache[(repo_type, repo_id, revision)] = True, True
                self._repo_and_revision_exists_cache[(repo_type, repo_id, None)] = True, True
            return self._repo_and_revision_exists_cache[(repo_type, repo_id, revision)]

    def resolve_path(self, path: str) -> Tuple[str, str, Optional[str], str]:
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
            revision_with_namespace = unquote(path.split("/")[2]) if len(path.split("/")) > 2 else None
            path_in_repo_with_namespace = "/".join(path.split("/")[3:])
            repo_id_without_namespace = path.split("/")[0]
            revision_without_namespace = unquote(path.split("/")[1]) if len(path.split("/")) > 1 else None
            path_in_repo_without_namespace = "/".join(path.split("/")[2:])

            repo_exists, revision_exists = self._repo_and_revision_exists(
                repo_id_with_namespace,
                repo_type=repo_type,
                revision=revision_with_namespace,
            )
            if repo_exists:
                repo_id = repo_id_with_namespace
                if revision_exists:
                    revision = revision_with_namespace
                    path_in_repo = path_in_repo_with_namespace
                else:
                    revision = None
                    path_in_repo = f"{revision_with_namespace}/{path_in_repo_with_namespace}".rstrip("/")
            else:
                repo_exists, revision_exists = self._repo_and_revision_exists(
                    repo_id_without_namespace,
                    repo_type=repo_type,
                    revision=revision_without_namespace,
                )
                if repo_exists:
                    repo_id = repo_id_without_namespace
                    if revision_exists:
                        revision = revision_without_namespace
                        path_in_repo = path_in_repo_without_namespace
                    else:
                        revision = None
                        path_in_repo = f"{revision_without_namespace}/{path_in_repo_without_namespace}".rstrip("/")
                else:
                    raise FileNotFoundError(f"No such repository: '{repo_id_with_namespace}'")
        else:
            repo_exists, revision_exists = self._repo_and_revision_exists(path, repo_type=repo_type, revision=None)
            if repo_exists:
                repo_id = path
                revision = None
                path_in_repo = ""
            else:
                # can't list repositories at the namespace level
                raise NotImplementedError("Acces to repositories lists is not implemented.")

        return repo_type, repo_id, revision, path_in_repo

    def _resolve_path_with_revision(self, path: str, revision: Optional[str]) -> Tuple[str, str, str, str]:
        repo_type, repo_id, revision_in_path, path_in_repo = self.resolve_path(path)
        if revision is not None:
            if revision_in_path is not None and revision_in_path != revision:
                raise ValueError(
                    f'Revision specified in path ("{revision_in_path}") and in `revision` argument ("{revision}") are'
                    " not the same."
                )
        else:
            revision = revision_in_path or huggingface_hub.constants.DEFAULT_REVISION
        return repo_type, repo_id, revision, path_in_repo

    def _unresolve_path(self, repo_type: str, repo_id: str, revision: Optional[str], path_in_repo: str) -> str:
        path = huggingface_hub.constants.REPO_TYPES_URL_PREFIXES.get(repo_type, "") + repo_id + "/"
        if revision is not None:
            path += quote(revision, safe="") + "/"
        path += path_in_repo
        path = path.rstrip("/")
        return path

    def invalidate_cache(self, path=None):
        if not path:
            self.dircache.clear()
            self._repo_and_revision_exists_cache.clear()
        else:
            repo_type, repo_id, revision, path_in_repo = self.resolve_path(path)
            revision = revision if revision != huggingface_hub.constants.DEFAULT_REVISION else None
            path = self._unresolve_path(repo_type, repo_id, None, path_in_repo)
            path = self._strip_protocol(path)
            while path:
                self.dircache.pop(path, None)
                path = self._parent(path)

    def _open(
        self,
        path: str,
        mode: str = "rb",
        revision: Optional[str] = None,
        **kwargs,
    ):
        if mode == "ab":
            raise NotImplementedError("Appending to remote files is not yet supported.")
        path = self._strip_protocol(path)
        return HfFile(self, path, mode=mode, revision=revision, **kwargs)

    def _rm(self, path, revision: Optional[str] = None, **kwargs):
        path = self._strip_protocol(path)
        repo_type, repo_id, revision, path_in_repo = self._resolve_path_with_revision(path, revision)
        operations = [huggingface_hub.CommitOperationDelete(path_in_repo=path_in_repo)]
        commit_message = f"Delete {path}"
        self._api.create_commit(
            repo_id=repo_id,
            repo_type=repo_type,
            token=self.token,
            operations=operations,
            revision=revision,
            commit_message=kwargs.get("commit_message", commit_message),
            commit_description=kwargs.get("commit_description"),
        )
        self.invalidate_cache(path=self._unresolve_path(repo_type, repo_id, revision, path_in_repo))

    def rm(self, path, recursive=False, maxdepth=None, revision: Optional[str] = None, **kwargs):
        repo_type, repo_id, revision, path_in_repo = self._resolve_path_with_revision(path, revision)
        root_path = huggingface_hub.constants.REPO_TYPES_URL_PREFIXES.get(repo_type, "") + repo_id
        paths = self.expand_path(path, recursive=recursive, maxdepth=maxdepth, revision=revision)
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
            revision=revision,
            commit_message=kwargs.get("commit_message", commit_message),
            commit_description=kwargs.get("commit_description"),
        )
        self.invalidate_cache(path=self._unresolve_path(repo_type, repo_id, revision, path_in_repo))

    def ls(self, path, detail=True, refresh=False, revision: Optional[str] = None, **kwargs):
        path = self._strip_protocol(path)
        repo_type, repo_id, revision, path_in_repo = self._resolve_path_with_revision(path, revision)
        revision = revision if revision != huggingface_hub.constants.DEFAULT_REVISION else None
        path = self._unresolve_path(repo_type, repo_id, revision, path_in_repo)
        if path not in self.dircache or refresh:
            path_prefix = self._unresolve_path(repo_type, repo_id, revision, "") + "/"
            tree_iter = self._iter_tree(path, revision=revision)
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

    def _iter_tree(self, path: str, revision: Optional[str] = None):
        path = self._strip_protocol(path)
        repo_type, repo_id, revision, path_in_repo = self._resolve_path_with_revision(path, revision)
        revision = quote(revision, safe="") if revision is not None else huggingface_hub.constants.DEFAULT_REVISION
        path = (f"{self._api.endpoint}/api/{repo_type}s/{repo_id}/tree/{revision}/{path_in_repo}").rstrip("/")
        headers = self._api._build_hf_headers()
        yield from huggingface_hub.utils._pagination.paginate(path, params=None, headers=headers)

    def cp_file(self, path1, path2, revision: Optional[str] = None, **kwargs):
        repo_type1, repo_id1, revision1, path_in_repo1 = self._resolve_path_with_revision(path1, revision)
        path1 = self._strip_protocol(path1)
        repo_type2, repo_id2, revision2, path_in_repo2 = self._resolve_path_with_revision(path2, revision)
        path2 = self._strip_protocol(path2)

        same_repo = repo_type1 == repo_type2 and repo_id1 == repo_id2

        if not same_repo and revision is not None:
            raise ValueError("The `revision` argument is ambiguous when copying between two different repositories.")

        # TODO: Wait for https://github.com/huggingface/huggingface_hub/issues/1083 to be resolved to simplify this logic
        if same_repo and self.info(path1, revision=revision1)["lfs"] is not None:
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
                        "oid": self.info(path1, revision=revision1)["lfs"]["oid"],
                    }
                ],
                "deletedFiles": [],
            }
            revision2 = (
                quote(revision2, safe="") if revision is not None else huggingface_hub.constants.DEFAULT_REVISION
            )
            r = requests.post(
                f"{self.endpoint}/api/{repo_type1}s/{repo_id1}/commit/{revision2}",
                json=payload,
                headers=headers,
            )
            huggingface_hub.utils.hf_raise_for_status(r)
        else:
            with self.open(path1, "rb", revision=revision1) as f:
                content = f.read()
            commit_message = f"Copy {path1} to {path2}"
            self._api.upload_file(
                path_or_fileobj=content,
                path_in_repo=path_in_repo2,
                repo_id=repo_id2,
                token=self.token,
                repo_type=repo_type2,
                revision=revision2,
                commit_message=kwargs.get("commit_message", commit_message),
                commit_description=kwargs.get("commit_description"),
            )
        self.invalidate_cache(path=self._unresolve_path(repo_type1, repo_id1, revision1, path_in_repo1))
        self.invalidate_cache(path=self._unresolve_path(repo_type2, repo_id2, revision2, path_in_repo2))

    def modified(self, path, **kwargs):
        info = self.info(path, **kwargs)
        if info["type"] != "file":
            raise FileNotFoundError(path)
        return info["last_modified"]

    def info(self, path, **kwargs):
        path = self._strip_protocol(path)
        repo_type, repo_id, revision, path_in_repo = self.resolve_path(path)
        if not path_in_repo:
            return {"name": path, "size": None, "type": "directory"}
        return super().info(path, **kwargs)

    def expand_path(self, path, recursive=False, maxdepth=None, **kwargs):
        if maxdepth is not None and maxdepth < 1:
            raise ValueError("maxdepth must be at least 1")

        if isinstance(path, str):
            out = self.expand_path([path], recursive, maxdepth)
        else:
            out = set()
            path = [self._strip_protocol(p) for p in path]
            for p in path:
                if has_magic(p):
                    bit = set(self.glob(p))
                    out |= bit
                    if recursive:
                        out |= set(self.expand_path(list(bit), recursive=recursive, maxdepth=maxdepth, **kwargs))
                    continue
                elif recursive:
                    rec = set(self.find(p, maxdepth=maxdepth, withdirs=True, detail=False, **kwargs))
                    out |= rec
                if p not in out and (recursive is False or self.exists(p)):
                    # should only check once, for the root
                    out.add(p)
        if not out:
            raise FileNotFoundError(path)
        return list(sorted(out))


class HfFile(fsspec.spec.AbstractBufferedFile):
    def __init__(self, fs: HfFileSystem, path: str, revision: Optional[str] = None, **kwargs):
        super().__init__(fs, path, **kwargs)
        self.fs: HfFileSystem
        self.repo_type, self.repo_id, self.revision, self.path_in_repo = fs._resolve_path_with_revision(path, revision)

    def _fetch_range(self, start, end):
        headers = {
            "range": f"bytes={start}-{end - 1}",
            **self.fs._api._build_hf_headers(),
        }
        revision = (
            quote(self.revision, safe="") if self.revision is not None else huggingface_hub.constants.DEFAULT_REVISION
        )
        url = (
            f"{self.fs.endpoint}/{huggingface_hub.constants.REPO_TYPES_URL_PREFIXES.get(self.repo_type, '') + self.repo_id}/resolve/{revision}/{quote(self.path_in_repo, safe='')}"
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
                revision=self.revision,
                commit_message=self.kwargs.get("commit_message", commit_message),
                commit_description=self.kwargs.get("commit_description"),
            )
            os.remove(self.temp_file.name)
            self.fs.invalidate_cache(
                path=self.fs._unresolve_path(self.repo_type, self.repo_id, self.revision, self.path_in_repo)
            )
