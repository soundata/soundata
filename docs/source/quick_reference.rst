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

* 📺 : Youtube Links only

* ❌ : Not available

Tasks Codes: 

| :sel:`SEL` : Sound Event Localization 
| :sed:`SED` : Sound Event Detection
| :sec:`SEC` : Sound Event Classification
| :asc:`ASC` : Acoustic Scene Classification
| :ac:`AC` : Audio Captioning

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

Usecases
================

Tasks
^^^^^^

.. _sel:

:sel:`Sound Event Localization (SEL)`
   SEL involves determining the spatial location from where a sound originates within an environment. It goes beyond detection and classification to include the position in space relative to the listener or recording device.

.. _sed:

:sed:`Sound Event Detection (SED)`
   SED is concerned with identifying the presence and duration of sound events within an audio stream. It uses both weak labels (Tags) for presence and strong labels (Events) for temporal localization of sound events.

.. _sec:

:sec:`Sound Event Classification (SEC)`
   SEC categorizes sounds into predefined classes and involves analyzing audio to assign a category based on the type of sound event it contains, using Tags for the entire clip's duration.

.. _asc:

:asc:`Acoustic Scene Classification (ASC)`
   ASC classifies an entire audio stream into a scene category, characterizing the recording's environment. Tags are used to indicate the single acoustic scene represented in the clip.

.. _ac:

:ac:`Audio Captioning (AC)`
   AC involves generating a textual description of the sound events and context within an audio clip. It is similar to image captioning but for audio content.


Soundscapes
^^^^^^^^^^^

.. _urban-environment:

:urban:`URBAN`
   Urban environments are characterized by a blend of sounds from traffic, human activity, construction, and sometimes nature. Recordings in urban areas are often used to study noise pollution, city planning, or to create soundscapes for multimedia productions.

.. _environment-sounds:

:environment:`ENVIRONMENT`
   The spectrum of environmental sounds includes all the background noises found in various habitats. These auditory elements can be as diverse as the whisper of foliage in woodlands, the gentle flow of water in brooks, or the fierce gusts of wind sweeping through arid landscapes.
.. _machine-sounds:

:machine:`MACHINE`
   Machine sounds refer to the audio signatures of mechanical devices, such as engines, factory machinery, household appliances, and office equipment. These sounds are crucial for monitoring equipment performance, diagnosing faults, and designing sound-aware applications.

.. _bioacoustic-sounds:

:bioacoustic:`BIOACOUSTIC`
   Bioacoustic sounds are produced by biological organisms, like the vocalizations of animals and birds. Studying these sounds can provide insights into animal behavior, biodiversity, and ecosystem health.

.. _music-sounds:

:music:`MUSIC`
   Music sounds encompass the vast array of musical compositions, instruments, and the human voice as used in singing. These sounds are central to the entertainment industry, cultural studies, and music therapy.
