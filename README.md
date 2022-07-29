# `hffs`

`hffs` builds on [`huggingface_hub`](https://github.com/huggingface/huggingface_hub) and [`fsspec`](https://github.com/huggingface/huggingface_hub) to provide a convenient Python filesystem interface to 🤗 Hub.

## Examples

Locate and read a file from a 🤗 Hub repo:

```python
>>> import hffs
>>> fs = hffs.HfFileSystem("my-username/my-dataset-repo", repo_type="dataset")
>>> fs.ls("")
['.gitattributes', 'my-file.txt']
>>> with fs.open("my-file.txt", "r") as f:
...     f.read()
'Hello, world'
```

Write a file to the repo:

```python
>>> with fs.open("my-file-new.txt", "w") as f:
...     f.write("Hello, world1")
...     f.write("Hello, world2")
>>> fs.exists("my-file-new.txt")
True
>>> fs.du("my-file-new.txt")
26
```

Instantiation via `fsspec`:

```python
>>> import fsspec

# Instantiate a `hffs.HfFileSystem` object
>>> fs = fsspec.filesystem("hf://my-username/my-dataset-repo", repo_type="dataset")
>>> fs.ls("")
['.gitattributes', 'my-file.txt']

# Instantiate a `hffs.HfFileSystem` object and write a file to it
>>> with fsspec.open("hf://my-username/my-dataset-repo:/my-file-new.txt", repo_type="dataset"):
...     f.write("Hello, world1")
...     f.write("Hello, world2")
```

> **Note**: To be recognized as a `hffs` URL, the URL path passed to [`fsspec.open`](https://filesystem-spec.readthedocs.io/en/latest/api.html?highlight=open#fsspec.open) must adhere to the following scheme:
> ```
> hf://<repo_id>[@<revision>]:/<path/in/repo>
> ```

# Installation

```bash
pip install hffs
```

## Integrations

* [`pandas`](https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#reading-writing-remote-files)/[`dask`](https://docs.dask.org/en/stable/how-to/connect-to-remote-data.html)

```python
>>> import pandas as pd

>>> # Read a remote CSV file into a dataframe
>>> df = pd.read_csv("hf://my-username/my-dataset-repo:/train.csv", storage_options={"repo_type": "dataset"})

>>> # Write a dataframe to a remote CSV file
>>> df.to_csv("hf://my-username/my-dataset-repo:/test.csv", storage_options={"repo_type": "dataset"})
```

* [`datasets`](https://huggingface.co/docs/datasets/filesystems#load-and-save-your-datasets-using-your-cloud-storage-filesystem)

```python
>>> import datasets

>>> # Cache a (large) dataset inside a repo
>>> cache_dir = "hf://my-username/my-dataset-repo"
>>> builder = datasets.load_dataset_builder("path/to/local/loading_script/loading_script.py", cache_dir=cache_dir, storage_options={"repo_type": "dataset"})
>>> builder.download_and_prepare(file_format="parquet")

>>> # Stream the dataset from the repo
>>> dset = datasets.load_dataset("my-username/my-dataset-repo", split="train")
>>> # Process the examples
>>> for ex in dset:
...    ...
```
