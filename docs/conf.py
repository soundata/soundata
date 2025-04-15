# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import datetime
from docutils import nodes, utils
from docutils.parsers.rst import roles

sys.path.insert(0, os.path.abspath("./source/_ext"))
sys.path.insert(0, os.path.abspath("../"))

# -- Project information -----------------------------------------------------

project = "soundata"
year = datetime.datetime.utcnow().year
copyright = '2021-{}, Soundata development team'.format(year)
author = "The Soundata development team"


import importlib

soundata_version = importlib.import_module("soundata.version")

# The short X.Y version.
version = soundata_version.short_version
# The full version, including alpha/beta/rc tags.
release = soundata_version.version
# Show only copyright
show_authors = False


# -- Mock dependencies -------------------------------------------------------
autodoc_mock_imports = ["librosa", "numpy", "jams", "pandas", "pydub", "simpleaudio", "seaborn", "py7zr", "matplotlib"]


# # -- General configuration ---------------------------------------------------

# # Add any Sphinx extension module names here, as strings. They can be
# # extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# # ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_togglebutton",
    "sphinx.ext.extlinks",
    "sphinx_togglebutton",
    "docstring_mod",
]

# To shorten links of licenses and add to table
extlinks = {
    "tau2019sse": ("https://zenodo.org/record/2580091%s", "Custom%s"),
    "tau2019": ("https://zenodo.org/record/2589280%s", "Custom%s"),
    "tau2020": ("https://zenodo.org/record/3819968%s", "Custom%s"),
    "tau2022": ("https://zenodo.org/record/6337421%s", "Custom%s"),
    "tut": ("https://github.com/TUT-ARG/DCASE2017-baseline-system/blob/master/EULA.pdf%s", "Custom%s"),
}

intersphinx_mapping = {
    "np": ("https://numpy.org/doc/stable/", None),
    "jams": ("https://jams.readthedocs.io/en/stable/", None),
}

# Napoleon settings
# https://github.com/sphinx-contrib/napoleon/issues/2
napoleon_custom_sections = [
    ("Cached Properties", "Other Parameters")
]  # todo - when above issue is closed, update to say "cached properties"
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = True
napoleon_use_param = False
napoleon_use_rtype = False
napoleon_type_aliases = None
napoleon_attr_annotations = True


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix of source filenames.
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "source/example.rst",
]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = [
    "css/custom.css",
]
togglebutton_hint = "Click here to show example"
html_logo = "img/soundata.png"


def create_reference_role(node_id):
    def role_fn(name, rawtext, text, lineno, inliner, options={}, content=[]):
        text = utils.unescape(text)
        # Use the same class name as defined in the CSS file for the reference node
        class_name = name.lower()  # This should match the class name used in the CSS
        ref_node = nodes.reference(rawtext, text, refuri=f'#{node_id}', classes=[class_name])
        return [ref_node], []
    return role_fn

def setup(app):
    role_to_target = {
        'sed': 'sed',
        'sec': 'sec',
        'sel': 'sel',
        'asc': 'asc',
        'ac': 'ac',
        'urban': 'urban-environment',
        'environment': 'environment-sounds',
        'machine': 'machine-sounds',
        'bioacoustic': 'bioacoustic-sounds',
        'music': 'music-sounds',
    }
    
    for role_name, node_id in role_to_target.items():
        app.add_role(role_name, create_reference_role(node_id))