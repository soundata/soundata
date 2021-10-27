.. _faq:

###
FAQ
###

.. admonition:: How do I add a new loader?
    :class: dropdown

    Take a look at our :ref:`contributing` docs!

.. admonition:: How do I get access to a dataset if the download function says it’s not available?
    :class: dropdown

    We don't distribute data ourselves, so unfortunately it is up to you to find the data yourself. We strongly encourage you to favor datasets which are currently available.

.. admonition:: Can you send me the data for a dataset which is not available?
    :class: dropdown

    Sorry, we do not host or distribute datasets.


.. admonition:: What is the canonical version of a loader?
    :class: dropdown

    The ``canonical`` version of a loader is the source version of a dataset, i.e. the version that you get directly from the creators of the dataset or similar oficial source.


.. admonition:: How do I request a new dataset?
    :class: dropdown

    Open an issue_ and tag it with the "New Loader" label.

    .. _issue: https://github.com/soundata/soundata/issues

.. admonition:: What do I do if my data fails validation?
    :class: dropdown

    Very often, data fails vaildation because of how the files are named or how the folder is structured. If this is the case, try renaming/reorganizing your data to match what soundata expects. If your data fails validation because of the checksums, this means that you are using data which is different from what most people are using, and you should try to get the more common dataset version, for example by using the data loader's download function.

.. admonition:: How do you choose the data that is used to create the checksums?
    :class: dropdown

    Whenever possible, the data downloaded using :code:`.download()` is the same data used to create the checksums. If this isn't possible, we did our best to get the data from the original source (the dataset creator) in order to create the checksum. If this is again not possible, we found as many versions of the data as we could from different users of the dataset, computed checksums on all of them and used the version which was the most common amongst them.


.. admonition:: Does soundata provide data loaders for pytorch/Tensorflow?
    :class: dropdown

    Not yet, but we plan to include this functionality soon. For now, check the examples of Tensorflow generators in :ref:`tutorial`

.. admonition:: A download link is broken for a loader's :code:`.download()` function. What do I do?
    :class: dropdown

    Please open an issue_ and tag it with the "broken link" label.

    .. _issue: https://github.com/soundata/soundata/issues

.. admonition:: Why the name, soundata?
    :class: dropdown

    soundata = sound + data, and the library was built for working with audio data.

.. admonition:: If I find a mistake in an annotation, should I fix it in the loader?
    :class: dropdown

    Please do not. All datasets have "mistakes", and we do not want to create another version of each dataset ourselves. The loaders should load the data as released. After that, it's up to the user what they want to do with it. If you are in doubt, open an issue_ and discuss it with the community.


.. admonition:: Does soundata support data which lives off-disk?
    :class: dropdown

    Yes. While the simple useage of soundata assumes that data lives on-disk, it can be used for off-disk data as well.
    See :ref:`Remote Data Example` for details.



