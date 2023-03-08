import datetime
import time
import unittest
from typing import Dict
from uuid import uuid4
import fsspec
import pytest
from fsspec.spec import AbstractBufferedFile

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
        self.hffs = HfFileSystem(self.repo_id, endpoint=ENDPOINT_STAGING, token=TOKEN, repo_type=self.repo_type)
        self.api = self._fs._api

        # Create dumb repo
        self.api.create_repo(self.repo_id, repo_type=self.repo_type, private=True)
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
        self.assertEqual(sorted(self.hffs.glob("*")), sorted([".gitattributes", "data"]))

    def test_file_type(self):
        self.assertTrue(self.hffs.isdir("data") and not self.hffs.isdir(".gitattributes"))
        self.assertTrue(self.hffs.isfile("data/text_data.txt") and not self.hffs.isfile("data"))

    def test_remove_file(self):
        self.hffs.rm_file("data/text_data.txt")
        self.assertEqual(self.hffs.glob("data/*"), ["data/binary_data.bin"])

    def test_remove_directory(self):
        self.hffs.rm("data", recursive=True)
        self.assertNotIn("data", self.hffs.ls(""))

    def test_read_file(self):
        with self.hffs.open("data/text_data.txt", "r") as f:
            self.assertEqual(f.read(), "dummy text data")

    def test_write_file(self):
        data = "new text data"
        with self.hffs.open("data/new_text_data.txt", "w") as f:
            f.write(data)
        self.assertIn("data/new_text_data.txt", self.hffs.glob("data/*"))
        with self.hffs.open("data/new_text_data.txt", "r") as f:
            self.assertEqual(f.read(), data)

    def test_write_file_multiple_chunks(self):
        data = "new text data big" * AbstractBufferedFile.DEFAULT_BLOCK_SIZE
        with self.hffs.open("data/new_text_data_big.txt", "w") as f:
            for _ in range(2):
                f.write(data)

        self.assertIn("data/new_text_data_big.txt", self.hffs.glob("data/*"))
        with self.hffs.open("data/new_text_data_big.txt", "r") as f:
            for _ in range(2):
                self.assertEqual(f.read(len(data)), data)

    @unittest.skip("Not implemented yet")
    def test_append_file(self):
        with self.hffs.open("data/text_data.txt", "a") as f:
            f.write(" appended text")

        with self.hffs.open("data/text_data.txt", "r") as f:
            self.assertEqual(f.read(), "dummy text data appended text")

    def test_copy_file(self):
        # Non-LFS file
        self.assertIsNone(self.hffs.info("data/text_data.txt")["lfs"])
        self.hffs.cp_file("data/text_data.txt", "data/text_data_copy.txt")
        with self.hffs.open("data/text_data_copy.txt", "r") as f:
            self.assertEqual(f.read(), "dummy text data")
        self.assertIsNone(self.hffs.info("data/text_data_copy.txt")["lfs"])
        # LFS file
        self.assertIsNotNone(self.hffs.info("data/binary_data.bin")["lfs"])
        self.hffs.cp_file("data/binary_data.bin", "data/binary_data_copy.bin")
        with self.hffs.open("data/binary_data_copy.bin", "rb") as f:
            self.assertEqual(f.read(), b"dummy binary data")
        self.assertIsNotNone(self.hffs.info("data/binary_data_copy.bin")["lfs"])

    def test_modified_time(self):
        self.assertIsInstance(self.hffs.modified("data/text_data.txt"), datetime.datetime)
        # should fail on a non-existing file/directory
        with self.assertRaises(FileNotFoundError):
            self.hffs.modified("data/not_existing_file.txt")
        # should fail on a directory
        with self.assertRaises(FileNotFoundError):
            self.hffs.modified("data")

    def test_initialize_from_fsspec(self):
        fs, _, paths = fsspec.get_fs_token_paths(
            f"hf://{self.repo_type}/{self.repo_id}:/data/text_data.txt",
            storage_options={
                "endpoint": ENDPOINT_STAGING,
                "token": TOKEN,
                "revision": "test-branch",
            },
        )
        self.assertIsInstance(fs, HfFileSystem)
        self.assertEqual(fs._api.endpoint, ENDPOINT_STAGING)
        self.assertEqual(fs.repo_id, self.repo_id)
        self.assertEqual(fs.token, TOKEN)
        self.assertEqual(fs.repo_type, self.repo_type)
        self.assertEqual(fs.revision, "test-branch")
        self.assertEqual(paths, ["data/text_data.txt"])

        fs, _, _ = fsspec.get_fs_token_paths(f"hf://{self.repo_id}:/data/text_data.txt")
        self.assertEqual(fs.repo_id, self.repo_id)
        self.assertEqual(fs.repo_type, "model")


@pytest.mark.parametrize(
    "path,expected",
    [
        # Repo type "model" by default
        ("gpt2", {"repo_id": "gpt2", "repo_type": "model"}),
        ("hf://gpt2", {"repo_id": "gpt2", "repo_type": "model"}),
        # Parse without protocol
        ("datasets/username/my_dataset", {"repo_id": "username/my_dataset", "repo_type": "dataset"}),
        # Parse with hf:// protocol
        ("hf://datasets/username/my_dataset", {"repo_id": "username/my_dataset", "repo_type": "dataset"}),
        # Parse with revision
        (
            "hf://datasets/username/my_dataset@0123456789",
            {"repo_id": "username/my_dataset", "repo_type": "dataset", "revision": "0123456789"},
        ),
        # Parse canonical models (no namespace)
        ("gpt2", {"repo_id": "gpt2", "repo_type": "model"}),
        ("hf://gpt2", {"repo_id": "gpt2", "repo_type": "model"}),
        # Canonical datasets are not parsed correctly by huggingface_hub yet. They are processed separately by hffs.
        ("datasets/squad", {"repo_id": "squad", "repo_type": "dataset"}),
        ("hf://datasets/squad", {"repo_id": "squad", "repo_type": "dataset"}),
    ],
)
def test_parse_hffs_path_success(path: str, expected: Dict[str, str]) -> None:
    assert HfFileSystem._get_kwargs_from_urls(path) == expected
