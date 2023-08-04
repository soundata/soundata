.. _datasets:


.. toctree::
   :maxdepth: 1
   :titlesonly:


##################################
Supported Datasets and Annotations
##################################

This table is provided as a guide for users to select appropriate datasets. The
list of annotations omits some metadata for brevity, and we document the dataset's
primary annotations only.

"Downloadable" possible values:

* ✅ : Freely downloadable

* 🔑 : Available upon request

* 📺 : Youtube Links only

* ❌ : Not available


Find the API documentation for each of the below datasets in :ref:`api`.

.. include:: table.rst

Annotation Types
================

The table above provides annotation types as a guide for choosing appropriate datasets.
Here we provide a rough guide to the types in this table, but we **strongly recommend** reading
the dataset specific documentation to ensure the data is as you expect. To see how these annotation
types are implemented in Soundata see :ref:`annotations`.


.. _tags:

Tags
^^^^^
One or more ``string labels`` with corresponding ``confidence values``. Tags do not have start or end times,
and span the full duration of the clip. Tags are used to represent annotations for:
* Acoustic Scene Classification (ASC)
* Sound Event Classification (SEC)
* Sound Event Detection (SED) - weak labels

When every Tags annotation in a dataset contains exactly one label, it is typically a ``multi-class`` problem.
When Tags annotations contain varying numbers of labels (including 0), it is typically a ``multi-label`` problem.

.. _events:

Events
^^^^^^
Sound events with a ``start time``, ``end time``, ``label``, and ``confidence``. Events are used to represent annotations for:
* Sound Event Detection (SED) - strong labels
