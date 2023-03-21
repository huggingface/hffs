# `hffs`

<a href="https://github.com/huggingface/hffs/actions/workflows/ci.yml?query=branch%3Amain"><img alt="Build" src="https://github.com/huggingface/hffs/actions/workflows/ci.yml/badge.svg?branch=main"></a>
<a href="https://github.com/huggingface/hffs/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/hffs.svg"></a>
<a href="https://github.com/huggingface/hffs"><img alt="Supported Python versions" src="https://img.shields.io/pypi/pyversions/hffs.svg"></a>
<a href="https://huggingface.co/docs/hffs/index"><img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/hffs/index.svg?down_color=red&down_message=offline&up_message=online&label=doc"></a>

`hffs` builds on [`huggingface_hub`](https://github.com/huggingface/huggingface_hub) and [`fsspec`](https://github.com/fsspec/filesystem_spec) to provide a convenient Python filesystem interface to 🤗 Hub.

## Basic usage

Locate and read a file from a 🤗 Hub repo:

```python
>>> import hffs
>>> fs = hffs.HfFileSystem()
>>> fs.ls("datasets/my-username/my-dataset-repo")
['.gitattributes', 'my-file.txt']
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
['.gitattributes', 'config.json', 'pytorch_model.bin']

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
