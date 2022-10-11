import time
import unittest

import fsspec
from fsspec.spec import AbstractBufferedFile
from huggingface_hub import HfApi

from hffs import HfFileSystem


USER = "__DUMMY_TRANSFORMERS_USER__"
# Not critical, only usable on the sandboxed CI instance.
TOKEN = "hf_94wBhPGp6KrrTH3KDchhKpRxZwd6dmHWLL"
ENDPOINT_STAGING = "https://hub-ci.huggingface.co"


class HfFileSystemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._api = HfApi(endpoint=ENDPOINT_STAGING)
        cls._api.set_access_token(TOKEN)
        cls._endpoint = ENDPOINT_STAGING
        cls._token = TOKEN

        if HfFileSystem.protocol not in fsspec.available_protocols():
            fsspec.register_implementation(HfFileSystem.protocol, HfFileSystem)

    @classmethod
    def tearDownClass(cls):
        cls._api.unset_access_token()

    def setUp(self):
        repo_name = f"repo_data-{int(time.time() * 10e3)}"
        repo_id = f"{USER}/{repo_name}"
        self._api.create_repo(
            repo_id,
            token=self._token,
            repo_type="dataset",
            private=True,
        )
        self._api.upload_file(
            path_or_fileobj="dummy text data".encode("utf-8"),
            path_in_repo="data/text_data.txt",
            repo_id=repo_id,
            token=TOKEN,
            repo_type="dataset",
        )
        self._api.upload_file(
            path_or_fileobj=b"dummy binary data",
            path_in_repo="data/binary_data.bin",
            repo_id=repo_id,
            token=TOKEN,
            repo_type="dataset",
        )
        self.repo_id = repo_id
        self.repo_type = "dataset"

    def tearDown(self):
        self._api.delete_repo(self.repo_id, token=self._token, repo_type=self.repo_type)

    def test_glob(self):
        hffs = HfFileSystem(self.repo_id, endpoint=self._endpoint, token=self._token, repo_type=self.repo_type)
        self.assertEqual(sorted(hffs.glob("*")), sorted([".gitattributes", "data"]))

    def test_file_type(self):
        hffs = HfFileSystem(self.repo_id, endpoint=self._endpoint, token=self._token, repo_type=self.repo_type)
        self.assertTrue(hffs.isdir("data") and not hffs.isdir(".gitattributes"))
        self.assertTrue(hffs.isfile("data/text_data.txt") and not hffs.isfile("data"))

    def test_remove_file(self):
        hffs = HfFileSystem(self.repo_id, endpoint=self._endpoint, token=self._token, repo_type=self.repo_type)
        hffs.rm_file("data/text_data.txt")
        self.assertEqual(hffs.glob("data/*"), ["data/binary_data.bin"])

    def test_remove_directory(self):
        hffs = HfFileSystem(self.repo_id, endpoint=self._endpoint, token=self._token, repo_type=self.repo_type)
        hffs.rm("data", recursive=True)
        self.assertNotIn("data", hffs.ls(""))

    def test_read_file(self):
        hffs = HfFileSystem(self.repo_id, endpoint=self._endpoint, token=self._token, repo_type=self.repo_type)
        with hffs.open("data/text_data.txt", "r") as f:
            self.assertEqual(f.read(), "dummy text data")

    def test_write_file(self):
        hffs = HfFileSystem(self.repo_id, endpoint=self._endpoint, token=self._token, repo_type=self.repo_type)
        data = "new text data"
        with hffs.open("data/new_text_data.txt", "w") as f:
            f.write(data)
        self.assertIn("data/new_text_data.txt", hffs.glob("data/*"))
        with hffs.open("data/new_text_data.txt", "r") as f:
            self.assertEqual(f.read(), data)

    def test_write_file_multiple_chunks(self):
        hffs = HfFileSystem(self.repo_id, endpoint=self._endpoint, token=self._token, repo_type=self.repo_type)
        data = "new text data big" * AbstractBufferedFile.DEFAULT_BLOCK_SIZE
        with hffs.open("data/new_text_data_big.txt", "w") as f:
            for _ in range(2):
                f.write(data)

        self.assertIn("data/new_text_data_big.txt", hffs.glob("data/*"))
        with hffs.open("data/new_text_data_big.txt", "r") as f:
            for _ in range(2):
                self.assertEqual(f.read(len(data)), data)

    @unittest.skip("Not implemented yet")
    def test_append_file(self):
        hffs = HfFileSystem(self.repo_id, endpoint=self._endpoint, token=self._token, repo_type=self.repo_type)
        with hffs.open("data/text_data.txt", "a") as f:
            f.write(" appended text")

        with hffs.open("data/text_data.txt", "r") as f:
            self.assertEqual(f.read(), "dummy text data appended text")

    def test_copy_file(self):
        hffs = HfFileSystem(self.repo_id, endpoint=self._endpoint, token=self._token, repo_type=self.repo_type)
        # Non-LFS file
        self.assertIsNone(hffs.info("data/text_data.txt")["lfs"])
        hffs.cp_file("data/text_data.txt", "data/text_data_copy.txt")
        with hffs.open("data/text_data_copy.txt", "r") as f:
            self.assertEqual(f.read(), "dummy text data")
        self.assertIsNone(hffs.info("data/text_data_copy.txt")["lfs"])
        # LFS file
        self.assertIsNotNone(hffs.info("data/binary_data.bin")["lfs"])
        hffs.cp_file("data/binary_data.bin", "data/binary_data_copy.bin")
        with hffs.open("data/binary_data_copy.bin", "rb") as f:
            self.assertEqual(f.read(), b"dummy binary data")
        self.assertIsNotNone(hffs.info("data/binary_data_copy.bin")["lfs"])

    def test_initialize_from_fsspec(self):
        fs, _, paths = fsspec.get_fs_token_paths(
            f"hf://{self.repo_id}:/data/text_data.txt",
            storage_options={
                "endpoint": self._endpoint,
                "token": self._token,
                "repo_type": self.repo_type,
                "revision": "test-branch",
            },
        )
        self.assertIsInstance(fs, HfFileSystem)
        self.assertEqual(fs._api.endpoint, self._endpoint)
        self.assertEqual(fs.repo_id, self.repo_id)
        self.assertEqual(fs.token, self._token)
        self.assertEqual(fs.repo_type, self.repo_type)
        self.assertEqual(fs.revision, "test-branch")
        self.assertEqual(paths, ["data/text_data.txt"])
