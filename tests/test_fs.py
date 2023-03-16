import datetime
import unittest
from typing import Optional
from unittest.mock import patch
from uuid import uuid4

import fsspec
import huggingface_hub
import pytest

from hffs import HfFileSystem


USER = "__DUMMY_TRANSFORMERS_USER__"
# Not critical, only usable on the sandboxed CI instance.
TOKEN = "hf_94wBhPGp6KrrTH3KDchhKpRxZwd6dmHWLL"
ENDPOINT_STAGING = "https://hub-ci.huggingface.co"


class HfFileSystemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if HfFileSystem.protocol not in fsspec.available_protocols():
            fsspec.register_implementation(HfFileSystem.protocol, HfFileSystem)

    def setUp(self):
        self.repo_id = f"{USER}/hffs-test-repo-{uuid4()}"
        self.repo_type = "dataset"
        self.repo_prefix = huggingface_hub.constants.REPO_TYPES_URL_PREFIXES.get(self.repo_type, "")
        self.hf_path = self.repo_prefix + self.repo_id
        self.hffs = HfFileSystem(endpoint=ENDPOINT_STAGING, token=TOKEN)
        self.api = self.hffs._api

        # Create dumb repo
        self.api.create_repo(self.repo_id, repo_type=self.repo_type, private=False)
        self.api.upload_file(
            path_or_fileobj="dummy text data".encode("utf-8"),
            path_in_repo="data/text_data.txt",
            repo_id=self.repo_id,
            repo_type=self.repo_type,
        )
        self.api.upload_file(
            path_or_fileobj=b"dummy binary data",
            path_in_repo="data/binary_data.bin",
            repo_id=self.repo_id,
            repo_type=self.repo_type,
        )

    def tearDown(self):
        self.api.delete_repo(self.repo_id, repo_type=self.repo_type)

    def test_glob(self):
        self.assertEqual(
            sorted(self.hffs.glob(self.hf_path + "/*")),
            sorted([self.hf_path + "/.gitattributes", self.hf_path + "/data"]),
        )

    def test_file_type(self):
        self.assertTrue(
            self.hffs.isdir(self.hf_path + "/data") and not self.hffs.isdir(self.hf_path + "/.gitattributes")
        )
        self.assertTrue(
            self.hffs.isfile(self.hf_path + "/data/text_data.txt") and not self.hffs.isfile(self.hf_path + "/data")
        )

    def test_remove_file(self):
        self.hffs.rm_file(self.hf_path + "/data/text_data.txt")
        self.assertEqual(self.hffs.glob(self.hf_path + "/data/*"), [self.hf_path + "/data/binary_data.bin"])

    def test_remove_directory(self):
        self.hffs.rm(self.hf_path + "/data", recursive=True)
        self.assertNotIn(self.hf_path + "/data", self.hffs.ls(self.hf_path))

    def test_read_file(self):
        with self.hffs.open(self.hf_path + "/data/text_data.txt", "r") as f:
            self.assertEqual(f.read(), "dummy text data")

    def test_write_file(self):
        data = "new text data"
        with self.hffs.open(self.hf_path + "/data/new_text_data.txt", "w") as f:
            f.write(data)
        self.assertIn(self.hf_path + "/data/new_text_data.txt", self.hffs.glob(self.hf_path + "/data/*"))
        with self.hffs.open(self.hf_path + "/data/new_text_data.txt", "r") as f:
            self.assertEqual(f.read(), data)

    def test_write_file_multiple_chunks(self):
        # TODO: try with files between 10 and 50MB (as of 16 March 2023 I was getting 504 errors on hub-ci)
        data = "a" * (4 << 20)  # 4MB
        with self.hffs.open(self.hf_path + "/data/new_text_data_big.txt", "w") as f:
            for _ in range(2):  # 8MB in total
                f.write(data)

        self.assertIn(self.hf_path + "/data/new_text_data_big.txt", self.hffs.glob(self.hf_path + "/data/*"))
        with self.hffs.open(self.hf_path + "/data/new_text_data_big.txt", "r") as f:
            for _ in range(2):
                self.assertEqual(f.read(len(data)), data)

    @unittest.skip("Not implemented yet")
    def test_append_file(self):
        with self.hffs.open(self.hf_path + "/data/text_data.txt", "a") as f:
            f.write(" appended text")

        with self.hffs.open(self.hf_path + "/data/text_data.txt", "r") as f:
            self.assertEqual(f.read(), "dummy text data appended text")

    def test_copy_file(self):
        # Non-LFS file
        self.assertIsNone(self.hffs.info(self.hf_path + "/data/text_data.txt")["lfs"])
        self.hffs.cp_file(self.hf_path + "/data/text_data.txt", self.hf_path + "/data/text_data_copy.txt")
        with self.hffs.open(self.hf_path + "/data/text_data_copy.txt", "r") as f:
            self.assertEqual(f.read(), "dummy text data")
        self.assertIsNone(self.hffs.info(self.hf_path + "/data/text_data_copy.txt")["lfs"])
        # LFS file
        self.assertIsNotNone(self.hffs.info(self.hf_path + "/data/binary_data.bin")["lfs"])
        self.hffs.cp_file(self.hf_path + "/data/binary_data.bin", self.hf_path + "/data/binary_data_copy.bin")
        with self.hffs.open(self.hf_path + "/data/binary_data_copy.bin", "rb") as f:
            self.assertEqual(f.read(), b"dummy binary data")
        self.assertIsNotNone(self.hffs.info(self.hf_path + "/data/binary_data_copy.bin")["lfs"])

    def test_modified_time(self):
        self.assertIsInstance(self.hffs.modified(self.hf_path + "/data/text_data.txt"), datetime.datetime)
        # should fail on a non-existing file/directory
        with self.assertRaises(FileNotFoundError):
            self.hffs.modified(self.hf_path + "/data/not_existing_file.txt")
        # should fail on a directory
        with self.assertRaises(FileNotFoundError):
            self.hffs.modified(self.hf_path + "/data")

    def test_initialize_from_fsspec(self):
        fs, _, paths = fsspec.get_fs_token_paths(
            f"hf://{self.repo_type}s/{self.repo_id}/data/text_data.txt",
            storage_options={
                "endpoint": ENDPOINT_STAGING,
                "token": TOKEN,
                "revision": "test-branch",
            },
        )
        self.assertIsInstance(fs, HfFileSystem)
        self.assertEqual(fs._api.endpoint, ENDPOINT_STAGING)
        self.assertEqual(fs.token, TOKEN)
        self.assertEqual(fs.revision, "test-branch")
        self.assertEqual(paths, [self.hf_path + "/data/text_data.txt"])

        fs, _, paths = fsspec.get_fs_token_paths(f"hf://{self.repo_id}/data/text_data.txt")
        self.assertIsInstance(fs, HfFileSystem)
        self.assertEqual(paths, [f"{self.repo_id}/data/text_data.txt"])


@pytest.mark.parametrize("path_in_repo", ["", "foo"])
@pytest.mark.parametrize(
    "root_path,repo_type,repo_id",
    [
        # Parse without namespace
        ("gpt2", None, "gpt2"),
        ("datasets/squad", "dataset", "squad"),
        # Parse with namespace
        ("username/my_model", None, "username/my_model"),
        ("datasets/username/my_dataset", "dataset", "username/my_dataset"),
        # Parse with hf:// protocol
        ("hf://gpt2", None, "gpt2"),
        ("hf://datasets/squad", "dataset", "squad"),
    ],
)
def test_resolve_repo_id(root_path: str, repo_type: Optional[str], repo_id: str, path_in_repo: str) -> None:
    fs = HfFileSystem()
    path = root_path + "/" + path_in_repo if path_in_repo else root_path

    def mock_repo_info(repo_id: str, *, repo_type: str, **kwargs):
        if repo_id not in ["gpt2", "squad", "username/my_dataset", "username/my_model"]:
            raise huggingface_hub.utils.RepositoryNotFoundError(repo_id)

    with patch.object(fs._api, "repo_info", mock_repo_info):
        assert fs._resolve_repo_id(path) == (repo_type, repo_id, path_in_repo)


@pytest.mark.parametrize("not_supported_path", ["", "datasets", "hf://", "hf://datasets"])
def test_list_repositories(not_supported_path):
    fs = HfFileSystem()
    with pytest.raises(NotImplementedError):
        fs.ls(not_supported_path)
