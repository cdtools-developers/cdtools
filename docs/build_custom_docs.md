# Build custom docs

This is a short description how to build custom documentation with `sphinx` based on the exact version (or even commit) you are using. 

This requires a cloned version of `cdtools` from GitHub. See the installation guide for more information: 

https://cdtools-developers.github.io/cdtools/installation.html#option-2-installation-from-source


## Installation of Dependencies

First, ensure you have all necessary dependencies installed. You can do this using [`uv`](https://github.com/astral-sh/uv) or `pip`:


```sh
uv pip install ."[docs]"
# or, if you prefer pip:
pip install ."[docs]"
```

This will install your project along with the extra dependencies required for building the documentation.

**Note:** `uv` is a fast Python package installer and resolver, serving as a drop-in replacement for `pip` with improved performance. You can use either `pip` or `uv` as shown above.


## Checkout the version or commit

To ensure your documentation matches a specific version or commit of your codebase, use `git` to checkout the desired state. For example, to checkout a specific tag or commit:

```sh
git checkout <tag-or-commit-hash>
```

Replace `<tag-or-commit-hash>` with the version tag (e.g., `v1.2.3`) or the commit hash you want to use. This ensures the documentation is built for the exact code you are working with.

## Building the Documentation

To build the HTML documentation, run the following command from the root of your project:

```sh
uv run python -m sphinx -b html docs/source docs/_build/html/
```

This command tells Sphinx to build the documentation located in the `docs/source` directory and output the HTML files to `docs/_build/html/`.

You can then open the generated HTML files in your browser to view the documentation.


## Get back to the latest version of cdtools

To return to the latest version of your code, use:

```sh
git checkout master
```

This will switch your working directory back to the latest development branch.