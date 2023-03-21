# `hffs`

`hffs` builds on [`huggingface_hub`](https://github.com/huggingface/huggingface_hub) and [`fsspec`](https://github.com/fsspec/filesystem_spec) to provide a convenient Python filesystem interface to 🤗 Hub.

## Basic usage

Locate and read a file from a 🤗 Hub repo:

```python
>>> import hffs
>>> fs = hffs.HfFileSystem()
>>> fs.ls("datasets/my-username/my-dataset-repo", detail=False)
['datasets/my-username/my-dataset-repo/.gitattributes', 'datasets/my-username/my-dataset-repo/my-file.txt']
>>> with fs.open("datasets/my-username/my-dataset-repo/my-file.txt", "r") as f:
...     f.read()
'Hello, world'
```

Write a file to the repo:

```python
>>> with fs.open("datasets/my-username/my-dataset-repo/my-file-new.txt", "w") as f:
...     f.write("Hello, world1")
...     f.write("Hello, world2")
>>> fs.exists("datasets/my-username/my-dataset-repo/my-file-new.txt")
True
>>> fs.du("datasets/my-username/my-dataset-repo/my-file-new.txt")
26
```

Instantiation via `fsspec`:

```python
>>> import fsspec

>>> # Instantiate a `hffs.HfFileSystem` object
>>> fs = fsspec.filesystem("hf")
>>> fs.ls("my-username/my-model-repo")
['my-username/my-model-repo/.gitattributes', 'my-username/my-model-repo/config.json', 'my-username/my-model-repo/pytorch_model.bin']

>>> # Instantiate a `hffs.HfFileSystem` object and write a file to it
>>> with fsspec.open("hf://datasets/my-username/my-dataset-repo/my-file-new.txt"):
...     f.write("Hello, world1")
...     f.write("Hello, world2")
```

> **Note**: To be recognized as a `hffs` URL, the URL path passed to [`fsspec.open`](https://filesystem-spec.readthedocs.io/en/latest/api.html?highlight=open#fsspec.open) must adhere to the following scheme:
> ```
> hf://[<repo_type_prefix>]<repo_id>/<path/in/repo>
> ```

The prefix for datasets is "datasets/", the prefix for spaces is "spaces/" and models don't need a prefix in the URL.

## Installation

```bash
pip install hffs
```

## Usage examples

* [`pandas`](https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#reading-writing-remote-files)/[`dask`](https://docs.dask.org/en/stable/how-to/connect-to-remote-data.html)

```python
>>> import pandas as pd

>>> # Read a remote CSV file into a dataframe
>>> df = pd.read_csv("hf://datasets/my-username/my-dataset-repo/train.csv")

>>> # Write a dataframe to a remote CSV file
>>> df.to_csv("hf://datasets/my-username/my-dataset-repo/test.csv")
```

* [`datasets`](https://huggingface.co/docs/datasets/filesystems#load-and-save-your-datasets-using-your-cloud-storage-filesystem)

```python
>>> import datasets

>>> # Export a (large) dataset to a repo
>>> output_dir = "hf://datasets/my-username/my-dataset-repo"
>>> builder = datasets.load_dataset_builder("path/to/local/loading_script/loading_script.py")
>>> builder.download_and_prepare(output_dir, file_format="parquet")

>>> # Stream the dataset from the repo
>>> dset = datasets.load_dataset("my-username/my-dataset-repo", split="train", streaming=True)
>>> # Process the examples
>>> for ex in dset:
...    ...
```

* [`zarr`](https://zarr.readthedocs.io/en/stable/tutorial.html#io-with-fsspec)

```python
>>> import numpy as np
>>> import zarr

>>> embeddings = np.random.randn(50000, 1000).astype("float32")

>>> # Write an array to a repo acting as a remote zarr store
>>> with zarr.open_group("hf://my-username/my-model-repo/array-store", mode="w") as root:
...    foo = root.create_group("embeddings")
...    foobar = foo.zeros('experiment_0', shape=(50000, 1000), chunks=(10000, 1000), dtype='f4')
...    foobar[:] = embeddings

>>> # Read from a remote zarr store
>>> with zarr.open_group("hf://my-username/my-model-repo/array-store", mode="r") as root:
...    first_row = root["embeddings/experiment_0"][0]
```

* [`duckdb`](https://duckdb.org/docs/guides/python/filesystems)

```python
>>> import hffs
>>> import duckdb

>>> fs = hffs.HfFileSystem()
>>> duckdb.register_filesystem(fs)
>>> # Query a remote file and get the result as a dataframe
>>> df = duckdb.query("SELECT * FROM 'hf://datasets/my-username/my-dataset-repo/data.parquet' LIMIT 10").df()
```

## Authentication

To write to your repotitories or access your private repositorories; you can login by running

```bash
huggingface-cli login
```

Or pass a token (from your [HF settings](https://huggingface.co/settings/tokens)) to

```python
>>> import hffs
>>> fs = hffs.HfFileSystem(token=token)
```

or as `storage_options`:

```python
>>> storage_options = {"token": token}
>>> df = pd.read_csv("hf://datasets/my-username/my-dataset-repo/train.csv", storage_options=storage_options)
```
