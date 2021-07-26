soundata
=======

.. toctree::
   :maxdepth: 1
   :titlesonly:



``soundata`` is an open-source Python library that provides tools for loading and working with sound datasets, including tools for:

 * Downloading datasets to a common location and format
 * Validating that the files for a dataset are all present and valid
 * Loading annotation files to a common format
 * Parsing clip level metadata for detailed evaluations.


.. code-block::

    pip install soundata


For more details on how to use the library see the :ref:`tutorial`.


Citing soundata
--------------

If you are using the library for your work, please cite the version you used as indexed at Zenodo:

.. image:: https://zenodo.org/badge/343599660.svg
   :target: https://zenodo.org/badge/latestdoi/343599660

If you refer to soundata's design principles, motivation etc., please cite the following
`paper <https://soundata.pdf>`_  [#]_:

.. [#] soundata paper reference (TBA)

When working with datasets, please cite the version of ``soundata`` that you are using (given by the ``DOI`` above)
**AND** include the reference of the dataset, which can be found in the respective dataset loader using the ``cite()`` method.


Contributing to soundata
-----------------------

We welcome contributions to this library, especially new datasets.
Please see :ref:`contributing` for guidelines.

- `Issue Tracker <https://github.com/soundata/soundata/issues>`_
- `Source Code <https://github.com/soundata/soundata>`_


.. toctree::
   :caption: Get Started
   :maxdepth: 1


   source/overview
   source/quick_reference
   source/tutorial

.. toctree::
   :caption: API documentation
   :maxdepth: 1

   source/soundata

.. toctree::
   :caption: Further Information
   :maxdepth: 1

   source/contributing
   source/faq


 