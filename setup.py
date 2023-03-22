# Lint as: python3
"""HuggingFace Filesystem is an interface to huggingface.co repositories.

Note:

   VERSION needs to be formatted following the MAJOR.MINOR.PATCH convention
   (we need to follow this convention to be able to retrieve versioned scripts)

Simple check list for release from AllenNLP repo: https://github.com/allenai/allennlp/blob/master/setup.py

Steps to make a release:

0. Prerequisites:
   - Dependencies:
     - twine: `pip install twine`
   - Create an account in (and join the 'hffs' project):
     - PyPI: https://pypi.org/
     - Test PyPI: https://test.pypi.org/

1. Create the release branch from main branch:
     ```
     git checkout main
     git pull upstream main
     git checkout -b release-VERSION
     ```
2. Change the version to the release VERSION in:
   - __init__.py
   - setup.py

3. Commit these changes, push and create a Pull Request:
     ```
     git add -u
     git commit -m "Release: VERSION"
     git push upstream release-VERSION
     ```
   - Go to: https://github.com/huggingface/hffs/pull/new/release
   - Create pull request

4. From your local release branch, build both the sources and the wheel. Do not change anything in setup.py between
   creating the wheel and the source distribution (obviously).
   - First, delete any building directories that may exist from previous builds:
     - build
     - dist
   - From the top level directory, build the wheel and the sources:
       ```
       python setup.py bdist_wheel
       python setup.py sdist
       ```
   - You should now have a /dist directory with both .whl and .tar.gz source versions.

5. Check that everything looks correct by uploading the package to the test PyPI server:
     ```
     twine upload dist/* -r pypitest --repository-url=https://test.pypi.org/legacy/
     ```
   Check that you can install it in a virtualenv/notebook by running:
     ```
     pip install huggingface_hub fsspec aiohttp
     pip install -U tqdm
     pip install -i https://testpypi.python.org/pypi hffs
     ```

6. Upload the final version to the actual PyPI:
     ```
     twine upload dist/* -r pypi
     ```

7. Make the release on GitHub once everything is looking hunky-dory:
   - Merge the release Pull Request
   - Create a new release: https://github.com/huggingface/hffs/releases/new
   - Choose a tag: Introduce the new VERSION as tag, that will be created when you publish the release
     - Create new tag VERSION on publish
   - Release title: Introduce the new VERSION as well
   - Describe the release
     - Use "Generate release notes" button for automatic generation
   - Publish release

8. Set the dev version
   - Create the dev-version branch from the main branch:
       ```
       git checkout main
       git pull upstream main
       git branch -D dev-version
       git checkout -b dev-version
       ```
   - Change the version to X.X.X+1.dev0 (e.g. VERSION=1.18.3 -> 1.18.4.dev0) in:
     - __init__.py
     - setup.py
   - Commit these changes, push and create a Pull Request:
       ```
       git add -u
       git commit -m "Set dev version"
       git push upstream dev-version
       ```
     - Go to: https://github.com/huggingface/hffs/pull/new/dev-version
     - Create pull request
   - Merge the dev version Pull Request
"""


from setuptools import find_packages, setup


REQUIRED_PKGS = [
    "fsspec",
    "requests",
    "huggingface_hub>=0.13.0",
]


TESTS_REQUIRE = [
    "pytest",
]


QUALITY_REQUIRE = ["black~=23.1", "ruff>=0.0.241"]


EXTRAS_REQUIRE = {
    "dev": TESTS_REQUIRE + QUALITY_REQUIRE,
    "tests": TESTS_REQUIRE,
    "quality": QUALITY_REQUIRE,
}

setup(
    name="hffs",
    version="0.0.1.dev0",  # expected format is one of x.y.z.dev0, or x.y.z.rc1 or x.y.z (no to dashes, yes to dots)
    description="Filesystem interface over huggingface.co repositories",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="HuggingFace Inc.",
    author_email="datasets@huggingface.co",
    url="https://github.com/huggingface/hffs",
    download_url="https://github.com/huggingface/hffs/tags",
    license="Apache 2.0",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.7.0",
    install_requires=REQUIRED_PKGS,
    extras_require=EXTRAS_REQUIRE,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="models datasets machine learning huggingface filesystem",
    zip_safe=False,  # Required for mypy to find the py.typed file
)
