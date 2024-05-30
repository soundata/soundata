"""TAU Urban Acoustic Scenes 2019 Loader

.. admonition:: Dataset Info
    :class: dropdown

    **TAU Urban Acoustic Scenes 2019, Development, Leaderboard and Evaluation datasets**

    `Audio Research Group,
    Tampere University of Technology <http://arg.cs.tut.fi/>`_

    *Authors*

    * `Toni Heittola <http://www.cs.tut.fi/~heittolt/>`_
    * `Annamaria Mesaros <http://www.cs.tut.fi/~mesaros/>`_
    * `Tuomas Virtanen <http://www.cs.tut.fi/~tuomasv/>`_

    *Recording and annotation*

    * Henri Laakso
    * Ronal Bejarano Rodriguez
    * Toni Heittola


    *Links*

    * `Development dataset <https://zenodo.org/record/2589280>`_
    * `Leaderboard dataset <https://zenodo.org/record/2672993>`_
    * `Evaluation dataset <https://zenodo.org/record/3063822>`_

    *Dataset*

    TAU Urban Acoustic Scenes 2019 dataset consists of 10-seconds audio segments from 10 acoustic scenes:

    - Airport - `airport`
    - Indoor shopping mall - `shopping_mall`
    - Metro station - `metro_station`
    - Pedestrian street - `street_pedestrian`
    - Public square - `public_square`
    - Street with medium level of traffic - `street_traffic`
    - Travelling by a tram - `tram`
    - Travelling by a bus - `bus`
    - Travelling by an underground metro - `metro`
    - Urban park - `park`

    A detailed description of the data recording and annotation procedure is available in:

    .. code-block:: latex

        Annamaria Mesaros, Toni Heittola, and Tuomas Virtanen.
        "A multi-device dataset for urban acoustic scene classification",
        In Proceedings of the Detection and Classification of Acoustic
        Scenes and Events 2018 Workshop (DCASE2018), Surrey, UK, 2018.

    *Development dataset*

    Each acoustic scene has 1440 segments (240 minutes of audio). The dataset
    contains in total 40 hours of audio.

    *Evaluation dataset*

    The dataset contains in total 7200 segments (20 hours of audio).

    *Leaderboard dataset*

    The dataset contains in total 1200 segments (200 minutes of audio).

    The dataset was collected by Tampere University of Technology between
    05/2018 -11/2018. The data collection has received funding from
    the European Research Council under the `ERC <https://erc.europa.eu/1>`_
    Grant Agreement 637422 EVERYSOUND.


    *Preparation of the dataset*

    The dataset was recorded in 12 large European cities: Amsterdam, Barcelona,
    Helsinki, Lisbon, London, Lyon, Madrid, Milan, Prague, Paris, Stockholm,
    and Vienna. For all acoustic scenes, audio was captured in multiple
    locations: different streets, different parks, different shopping malls.
    In each location, multiple 2-3 minute long audio recordings were captured
    in a few slightly different positions (2-4) within the selected location.
    Collected audio material was cut into segments of 10 seconds length. 

    The equipment used for recording consists of a binaural `Soundman OKM II
    Klassik/studio A3 <http://www.soundman.de/en/products/>`_ electret in-ear
    microphone and a `Zoom F8
    <https://www.zoom.co.jp/products/handy-recorder/zoom-f8-multitrack-field-recorder>`_
    audio recorder using 48 kHz sampling rate and 24 bit resolution. During the
    recording, the microphones were worn by the recording person in the ears,
    and head movement was kept to minimum.

    Post-processing of the recorded audio involves aspects related to privacy
    of recorded individuals, and possible errors in the recording process. The
    material was screened for content, and segments containing close microphone
    conversation were eliminated. Some interferences from mobile phones are
    audible, but are considered part of real-world recording process.

    A subset of the dataset has been previously published as TUT Urban Acoustic
    Scenes 2018 Development dataset. Audio segment filenames are retained for
    the segments coming from this dataset.


    *Dataset statistics*

    The **development dataset** contains audio material from 10 cities, whereas
    the evaluation dataset (TAU Urban Acoustic Scenes 2019 evaluation) contains
    data from all 12 cities. The dataset is perfectly balanced at acoustic
    scene level, with very slight differences in the number of segments from
    each city.

    *Audio segments (Development dataset)*

    ===================  ==========  ==========  ==========  =========  ========= ==========  =========  =========  =========  ==========  ==========
    Scene class          Segments    Barcelona   Helsinki    Lisbon     London     Lyon       Milan      Paris      Prague     Stockholm   Vienna    
    ===================  ==========  ==========  ==========  =========  ========= ==========  =========  =========  =========  ==========  ==========
    Airport              1440        128         149         144        145        144        144        156        144        158         128
    Bus                  1440        144         144         144        144        144        144        144        144        144         144
    Metro                1440        141         144         144        146        144        144        144        144        145         144
    Metro station        1440        144         144         144        144        144        144        144        144        144         144
    Park                 1440        144         144         144        144        144        144        144        144        144         144
    Public square        1440        144         144         144        144        144        144        144        144        144         144
    Shopping mall        1440        144         144         144        144        144        144        144        144        144         144
    Street, pedestrian   1440        145         145         144        145        144        144        144        144        145         140
    Street, traffic      1440        144         144         144        144        144        144        144        144        144         144
    Tram                 1440        143         145         144        144        144        144        144        144        144         144
    **Total**            **14400**   **1421**    **1447**    **1440**   **1444**   **1440**   **1440**   **1452**   **1440**   **1456**    **1420**
    ===================  ==========  ==========  ==========  =========  ========= ==========  =========  =========  =========  ==========  ==========    

    *Audio segments (Recording locations)*

    ===================  ==========  ==========  ==========  =========  =========  =========  =========  =========  =========  ==========  ==========  
    Scene class          Locations   Barcelona   Helsinki    Lisbon     London     Lyon       Milan      Paris      Prague     Stockholm   Vienna    
    ===================  ==========  ==========  ==========  =========  =========  =========  =========  =========  =========  ==========  ==========  
    Airport              40          4           3           4          3          4          4          4          6          5           3         
    Bus                  71          4           4           11         7          7          7          11         10         6           4         
    Metro                67          3           5           11         4          9          8          9          10         4           4         
    Metro station        57          5           6           4          12         5          4          9          4          4           4         
    Park                 41          4           4           4          4          4          4          4          4          5           4         
    Public_square        43          4           4           4          4          5          4          4          6          4           4         
    Shopping mall        36          4           4           4          2          3          3          4          4          4           4         
    Street, pedestrian   46          7           4           4          4          4          5          5          5          4           4         
    Street, traffic      43          4           4           4          5          4          6          4          4          4           4         
    Tram                 70          4           4           6          9          7          11         9          11         5           4         
    **Total**            **514**     **43**      **42**      **56**     **54**     **52**     **56**     **63**     **65**     **45**      **39**    
    ===================  ==========  ==========  ==========  =========  =========  =========  =========  =========  =========  ==========  ==========  

    *Usage*

    The partitioning of the data was done based on the location of the original
    recordings. All segments recorded at the same location were included into a
    single subset - either **development dataset** or **evaluation dataset**.
    For each acoustic scene, 1440 segments were included in the development
    dataset provided here. Evaluation dataset is provided separately.

    *Training / test setup*

    A suggested training/test partitioning of the development set is provided
    in order to make results reported with this dataset uniform. The
    partitioning is done such that the segments recorded at the same location
    are included into the same subset - either training or testing. The
    partitioning is done aiming for a 70/30 ratio between the number of
    segments in training and test  subsets while taking into account recording
    locations, and selecting the closest available option. Audio segments
    coming from nine cities are used for training and all ten cities are used
    for testing (Milan is used only for testing). Since the dataset includes
    balanced amount of material from ten cities, this partitioning will leave a
    small subset of data from Milan unused in the training / test setup. This
    material can be used when using full dataset to train the system and
    testing it with evaluation dataset.

    The setup is provided with the dataset in the directory `evaluation_setup`. 

    *Statistics*

    ===================  =================  ==================  ================  =================  ==================  =================== 
    Scene class          Train / Segments   Train / Locations   Test / Segments   Test / Locations   Unused / Segments   Unused / Locations   
    ===================  =================  ==================  ================  =================  ==================  =================== 
    Airport              911                25                  421               12                 108                 3                   
    Bus                  928                46                  415               20                 97                  5                   
    Metro                902                41                  433               20                 105                 6                   
    Metro station        897                37                  435               17                 108                 3                   
    Park                 946                27                  386               11                 108                 3                   
    Public square        945                28                  387               12                 108                 3                   
    Shopping mall        896                24                  441               10                 103                 2                   
    Street, pedestrian   924                29                  429               14                 87                  3                   
    Street, traffic      942                27                  402               12                 96                  4                   
    Tram                 894                41                  436               21                 110                 8                   
    **Total**            **9185**           **325**             **4185**          **149**            **1030**            **40**              
    ===================  =================  ==================  ================  =================  ==================  =================== 


    *License*

    License permits free academic usage. Any commercial use is strictly
    prohibited. For commercial use, contact dataset authors.

        Copyright (c) 2019 Tampere University and its licensors
        All rights reserved.
        Permission is hereby granted, without written agreement and without license or royalty
        fees, to use and copy the TAU Urban Acoustic Scenes 2019 (“Work”) described in this document
        and composed of audio and metadata. This grant is only for experimental and non-commercial
        purposes, provided that the copyright notice in its entirety appear in all copies of this Work,
        and the original source of this Work, (Audio Research Group at Tampere University of Technology),
        is acknowledged in any publication that reports research using this Work.
        Any commercial use of the Work or any part thereof is strictly prohibited.
        Commercial use include, but is not limited to:
        - selling or reproducing the Work
        - selling or distributing the results or content achieved by use of the Work
        - providing services by using the Work.

        IN NO EVENT SHALL TAMPERE UNIVERSITY OR ITS LICENSORS BE LIABLE TO ANY PARTY
        FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE
        OF THIS WORK AND ITS DOCUMENTATION, EVEN IF TAMPERE UNIVERSITY OR ITS
        LICENSORS HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

        TAMPERE UNIVERSITY AND ALL ITS LICENSORS SPECIFICALLY DISCLAIMS ANY
        WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
        FITNESS FOR A PARTICULAR PURPOSE. THE WORK PROVIDED HEREUNDER IS ON AN "AS IS" BASIS, AND
        THE TAMPERE UNIVERSITY HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT,
        UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
"""

import os
from typing import BinaryIO, Optional, TextIO, Tuple

import librosa
import numpy as np
import csv

from soundata import download_utils, jams_utils, core, annotations, io


BIBTEX = """
@inproceedings{Mesaros:DCASE:18,
    Address = {Surrey, UK},
    Author = {Mesaros, A. and Heittola, T. and Virtanen, T.},
    Booktitle = {Proceedings of the Detection and Classification of Acoustic
                 Scenes and Events 2018 Workshop (DCASE2018)},
    Month = {November},
    Pages = {9--13},
    Title = {A multi-device dataset for urban acoustic scene classification},
    Year = {2018}}
"""

INDEXES = {
    "default": "1.0",
    "test": "sample",
    "1.0": core.Index(
        filename="tau2019uas_index_1.0.json",
        url="https://zenodo.org/records/11176859/files/tau2019uas_index_1.0.json?download=1",
        checksum="b1d7af813507b4943540397c519c7a0b",
    ),
    "sample": core.Index(filename="tau2019uas_index_1.0_sample.json"),
}

REMOTES = {
    "development.audio.1": download_utils.RemoteFileMetadata(
        filename="TAU-urban-acoustic-scenes-2019-development.audio.1.zip",
        url="https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.1.zip?download=1",
        checksum="aca4ebfd9ed03d5f747d6ba8c24bc728",
    ),
    "development.audio.2": download_utils.RemoteFileMetadata(
        filename="TAU-urban-acoustic-scenes-2019-development.audio.2.zip",
        url="https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.2.zip?download=1",
        checksum="c4f170408ce77c8c70c532bf268d7be0",
    ),
    "development.audio.3": download_utils.RemoteFileMetadata(
        filename="TAU-urban-acoustic-scenes-2019-development.audio.3.zip",
        url="https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.3.zip?download=1",
        checksum="c7214a07211f10f3250290d05e72c37e",
    ),
    "development.audio.4": download_utils.RemoteFileMetadata(
        filename="TAU-urban-acoustic-scenes-2019-development.audio.4.zip",
        url="https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.4.zip?download=1",
        checksum="a6a62110f6699cf4432072acb1dffda6",
    ),
    "development.audio.5": download_utils.RemoteFileMetadata(
        filename="TAU-urban-acoustic-scenes-2019-development.audio.5.zip",
        url="https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.5.zip?download=1",
        checksum="091a0b6d3c84b8e60e46940aa7d4a8a0",
    ),
    "development.audio.6": download_utils.RemoteFileMetadata(
        filename="TAU-urban-acoustic-scenes-2019-development.audio.6.zip",
        url="https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.6.zip?download=1",
        checksum="114f4ca13e074391b98a1cfd8140de65",
    ),
    "development.audio.7": download_utils.RemoteFileMetadata(
        filename="TAU-urban-acoustic-scenes-2019-development.audio.7.zip",
        url="https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.7.zip?download=1",
        checksum="5951dd2968f7a514e2afbe279c4f060d",
    ),
    "development.audio.8": download_utils.RemoteFileMetadata(
        filename="TAU-urban-acoustic-scenes-2019-development.audio.8.zip",
        url="https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.8.zip?download=1",
        checksum="b0b63dc95b327e1509857c8d8a663cc3",
    ),
    "development.audio.9": download_utils.RemoteFileMetadata(
        filename="TAU-urban-acoustic-scenes-2019-development.audio.9.zip",
        url="https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.9.zip?download=1",
        checksum="3c32a693a6b111ffb957be3c1dd22e9b",
    ),
    "development.audio.10": download_utils.RemoteFileMetadata(
        filename="TAU-urban-acoustic-scenes-2019-development.audio.10.zip",
        url="https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.10.zip?download=1",
        checksum="0ffbf60006da520cc761fb74c878b98b",
    ),
    "development.audio.11": download_utils.RemoteFileMetadata(
        filename="TAU-urban-acoustic-scenes-2019-development.audio.11.zip",
        url="https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.11.zip?download=1",
        checksum="599055d93b4c11057c29be2df54538d4",
    ),
    "development.audio.12": download_utils.RemoteFileMetadata(
        filename="TAU-urban-acoustic-scenes-2019-development.audio.12.zip",
        url="https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.12.zip?download=1",
        checksum="98b8d162ff3665695c4c910e6c372cc8",
    ),
    "development.audio.13": download_utils.RemoteFileMetadata(
        filename="TAU-urban-acoustic-scenes-2019-development.audio.13.zip",
        url="https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.13.zip?download=1",
        checksum="a356c08b1a5a21d433eba37ef87587f4",
    ),
    "development.audio.14": download_utils.RemoteFileMetadata(
        filename="TAU-urban-acoustic-scenes-2019-development.audio.14.zip",
        url="https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.14.zip?download=1",
        checksum="f8969771e7faf7dd471d1cf78b0cf011",
    ),
    "development.audio.15": download_utils.RemoteFileMetadata(
        filename="TAU-urban-acoustic-scenes-2019-development.audio.15.zip",
        url="https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.15.zip?download=1",
        checksum="4758c4b0fb7484faa632266e78850820",
    ),
    "development.audio.16": download_utils.RemoteFileMetadata(
        filename="TAU-urban-acoustic-scenes-2019-development.audio.16.zip",
        url="https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.16.zip?download=1",
        checksum="a18acad9ede8ea76574216feb887f0bc",
    ),
    "development.audio.17": download_utils.RemoteFileMetadata(
        filename="TAU-urban-acoustic-scenes-2019-development.audio.17.zip",
        url="https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.17.zip?download=1",
        checksum="1af7703484632f340da5c33662dc9632",
    ),
    "development.audio.18": download_utils.RemoteFileMetadata(
        filename="TAU-urban-acoustic-scenes-2019-development.audio.18.zip",
        url="https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.18.zip?download=1",
        checksum="b67402bf3e08f4da394a7c18756c0fd2",
    ),
    "development.audio.19": download_utils.RemoteFileMetadata(
        filename="TAU-urban-acoustic-scenes-2019-development.audio.19.zip",
        url="https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.19.zip?download=1",
        checksum="035db315f19106eb848b6f9b32bcc47c",
    ),
    "development.audio.20": download_utils.RemoteFileMetadata(
        filename="TAU-urban-acoustic-scenes-2019-development.audio.20.zip",
        url="https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.20.zip?download=1",
        checksum="9cb28c74911bf8a3eadcf53f50a5b5d6",
    ),
    "development.audio.21": download_utils.RemoteFileMetadata(
        filename="TAU-urban-acoustic-scenes-2019-development.audio.21.zip",
        url="https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.21.zip?download=1",
        checksum="0e44ed85c88ec036a9725b4dd1dfaea0",
    ),
    "development.doc": download_utils.RemoteFileMetadata(
        filename="TAU-urban-acoustic-scenes-2019-development.doc.zip",
        url="https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.doc.zip?download=1",
        checksum="1f6879544e80da70099a191613e7e51f",
    ),
    "development.meta": download_utils.RemoteFileMetadata(
        filename="TAU-urban-acoustic-scenes-2019-development.meta.zip",
        url="https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.meta.zip?download=1",
        checksum="09782f2097e4735687af73c44919329c",
    ),
    "evaluation.audio.1": download_utils.RemoteFileMetadata(
        filename="TAU-urban-acoustic-scenes-2019-evaluation.audio.1.zip",
        url="https://zenodo.org/record/3063822/files/TAU-urban-acoustic-scenes-2019-evaluation.audio.1.zip?download=1",
        checksum="3dfc50f8dc46f4a83a2e9cf2083d1c2a",
    ),
    "evaluation.audio.2": download_utils.RemoteFileMetadata(
        filename="TAU-urban-acoustic-scenes-2019-evaluation.audio.2.zip",
        url="https://zenodo.org/record/3063822/files/TAU-urban-acoustic-scenes-2019-evaluation.audio.2.zip?download=1",
        checksum="cc3d2a4b8e98ce0122317e401d0c6b7d",
    ),
    "evaluation.audio.3": download_utils.RemoteFileMetadata(
        filename="TAU-urban-acoustic-scenes-2019-evaluation.audio.3.zip",
        url="https://zenodo.org/record/3063822/files/TAU-urban-acoustic-scenes-2019-evaluation.audio.3.zip?download=1",
        checksum="a4815bdfd889a59f71c586cc834fc5e8",
    ),
    "evaluation.audio.4": download_utils.RemoteFileMetadata(
        filename="TAU-urban-acoustic-scenes-2019-evaluation.audio.4.zip",
        url="https://zenodo.org/record/3063822/files/TAU-urban-acoustic-scenes-2019-evaluation.audio.4.zip?download=1",
        checksum="d85f72ef7ae8a60b297da9e5bf478356",
    ),
    "evaluation.audio.5": download_utils.RemoteFileMetadata(
        filename="TAU-urban-acoustic-scenes-2019-evaluation.audio.5.zip",
        url="https://zenodo.org/record/3063822/files/TAU-urban-acoustic-scenes-2019-evaluation.audio.5.zip?download=1",
        checksum="73c84daf879db5cc4811808794e373de",
    ),
    "evaluation.audio.6": download_utils.RemoteFileMetadata(
        filename="TAU-urban-acoustic-scenes-2019-evaluation.audio.6.zip",
        url="https://zenodo.org/record/3063822/files/TAU-urban-acoustic-scenes-2019-evaluation.audio.6.zip?download=1",
        checksum="39d3cda72353ee9da88b78164350ff9f",
    ),
    "evaluation.audio.7": download_utils.RemoteFileMetadata(
        filename="TAU-urban-acoustic-scenes-2019-evaluation.audio.7.zip",
        url="https://zenodo.org/record/3063822/files/TAU-urban-acoustic-scenes-2019-evaluation.audio.7.zip?download=1",
        checksum="bd6fbf0d9324d1faa72968c162b574d7",
    ),
    "evaluation.audio.8": download_utils.RemoteFileMetadata(
        filename="TAU-urban-acoustic-scenes-2019-evaluation.audio.8.zip",
        url="https://zenodo.org/record/3063822/files/TAU-urban-acoustic-scenes-2019-evaluation.audio.8.zip?download=1",
        checksum="d7b4b62156f458865e2bd063b3da39e8",
    ),
    "evaluation.audio.9": download_utils.RemoteFileMetadata(
        filename="TAU-urban-acoustic-scenes-2019-evaluation.audio.9.zip",
        url="https://zenodo.org/record/3063822/files/TAU-urban-acoustic-scenes-2019-evaluation.audio.9.zip?download=1",
        checksum="7dbc037eca8d1234de8cd677853f72e4",
    ),
    "evaluation.audio.10": download_utils.RemoteFileMetadata(
        filename="TAU-urban-acoustic-scenes-2019-evaluation.audio.10.zip",
        url="https://zenodo.org/record/3063822/files/TAU-urban-acoustic-scenes-2019-evaluation.audio.10.zip?download=1",
        checksum="9a0b1e0d2647f6241d7b7c0649855cc7",
    ),
    "evaluation.audio.11": download_utils.RemoteFileMetadata(
        filename="TAU-urban-acoustic-scenes-2019-evaluation.audio.11.zip",
        url="https://zenodo.org/record/3063822/files/TAU-urban-acoustic-scenes-2019-evaluation.audio.11.zip?download=1",
        checksum="c2ae0b8d9270d964f8c1d8b5298fea72",
    ),
    "evaluation.doc": download_utils.RemoteFileMetadata(
        filename="TAU-urban-acoustic-scenes-2019-evaluation.doc.zip",
        url="https://zenodo.org/record/3063822/files/TAU-urban-acoustic-scenes-2019-evaluation.doc.zip?download=1",
        checksum="2fd4dc78299fc0d05212ca43dd89d922",
    ),
    "evaluation.meta": download_utils.RemoteFileMetadata(
        filename="TAU-urban-acoustic-scenes-2019-evaluation.meta.zip",
        url="https://zenodo.org/record/3063822/files/TAU-urban-acoustic-scenes-2019-evaluation.meta.zip?download=1",
        checksum="0b42d3c337b29d2efe50edd1e9496d7e",
    ),
    "leaderboard.audio.1": download_utils.RemoteFileMetadata(
        filename="TAU-urban-acoustic-scenes-2019-leaderboard.audio.1.zip",
        url="https://zenodo.org/record/2672993/files/TAU-urban-acoustic-scenes-2019-leaderboard.audio.1.zip?download=1",
        checksum="a5daa0df61c6fbc65b1e70f98d728ea3",
    ),
    "leaderboard.audio.2": download_utils.RemoteFileMetadata(
        filename="TAU-urban-acoustic-scenes-2019-leaderboard.audio.2.zip",
        url="https://zenodo.org/record/2672993/files/TAU-urban-acoustic-scenes-2019-leaderboard.audio.2.zip?download=1",
        checksum="c57c37a7ab6a32233583e39ec8cfafd5",
    ),
    "leaderboard.doc": download_utils.RemoteFileMetadata(
        filename="TAU-urban-acoustic-scenes-2019-leaderboard.doc.zip",
        url="https://zenodo.org/record/2672993/files/TAU-urban-acoustic-scenes-2019-leaderboard.doc.zip?download=1",
        checksum="826ede1a356e40ed6c80d873a0e10a70",
    ),
    "leaderboard.meta": download_utils.RemoteFileMetadata(
        filename="TAU-urban-acoustic-scenes-2019-leaderboard.meta.zip",
        url="https://zenodo.org/record/2672993/files/TAU-urban-acoustic-scenes-2019-leaderboard.meta.zip?download=1",
        checksum="fa3451868a2adf9d8a91882604a2d9b5",
    ),
}

LICENSE_INFO = """
License permits free a  cademic usage. Any commercial use is strictly prohibited. For commercial use, contact dataset authors.

    Copyright (c) 2019 Tampere University and its licensors
    All rights reserved.
    Permission is hereby granted, without written agreement and without license or royalty
    fees, to use and copy the TAU Urban Acoustic Scenes 2019 (“Work”) described in this document
    and composed of audio and metadata. This grant is only for experimental and non-commercial
    purposes, provided that the copyright notice in its entirety appear in all copies of this Work,
    and the original source of this Work, (Audio Research Group at Tampere University of Technology),
    is acknowledged in any publication that reports research using this Work.
    Any commercial use of the Work or any part thereof is strictly prohibited.
    Commercial use include, but is not limited to:
    - selling or reproducing the Work
    - selling or distributing the results or content achieved by use of the Work
    - providing services by using the Work.

    IN NO EVENT SHALL TAMPERE UNIVERSITY OR ITS LICENSORS BE LIABLE TO ANY PARTY
    FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE
    OF THIS WORK AND ITS DOCUMENTATION, EVEN IF TAMPERE UNIVERSITY OR ITS
    LICENSORS HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    TAMPERE UNIVERSITY AND ALL ITS LICENSORS SPECIFICALLY DISCLAIMS ANY
    WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
    FITNESS FOR A PARTICULAR PURPOSE. THE WORK PROVIDED HEREUNDER IS ON AN "AS IS" BASIS, AND
    THE TAMPERE UNIVERSITY HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT,
    UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
"""


class Clip(core.Clip):
    """TAU Urban Acoustic Scenes 2019 Clip class

    Args:
        clip_id (str): id of the clip

    Attributes:
        audio (np.ndarray, float): path to the audio file
        audio_path (str): path to the audio file
        city (str): city were the audio signal was recorded
        clip_id (str): clip id
        identifier (str): identifier present in the metadata
        split (str): subset the clip belongs to (for experiments):
            development (fold1, fold2, fold3, fold4), leaderboard or evaluation
        tags (soundata.annotations.Tags): tag (scene label) of the clip + confidence.
    """

    def __init__(self, clip_id, data_home, dataset_name, index, metadata):
        super().__init__(clip_id, data_home, dataset_name, index, metadata)

        self.audio_path = self.get_path("audio")

    @property
    def audio(self) -> Optional[Tuple[np.ndarray, float]]:
        """The clip's audio

        Returns:
            * np.ndarray - audio signal
            * float - sample rate

        """
        return load_audio(self.audio_path)

    @property
    def split(self):
        """The clip's split.

        Returns:
            * str - subset the clip belongs to (for experiments): development (fold1, fold2, fold3, fold4), leaderboard or evaluation

        """
        return self._clip_metadata.get("split")

    @property
    def tags(self):
        """The clip's tags.

        Returns:
            * annotations.Tags - tag (scene label) of the clip + confidence.

        """
        scene_label = self._clip_metadata.get("scene_label")
        if scene_label is None:
            return None
        else:
            return annotations.Tags([scene_label], "open", np.array([1.0]))

    @property
    def city(self):
        """The clip's city.

        Returns:
            * str - city were the audio signal was recorded

        """
        return self._clip_metadata.get("city")

    @property
    def identifier(self):
        """The clip's identifier.

        Returns:
            * str - identifier present in the metadata

        """
        return self._clip_metadata.get("identifier")

    def to_jams(self):
        """Get the clip's data in jams format

        Returns:
            jams.JAMS: the clip's data in jams format

        """
        return jams_utils.jams_converter(
            audio_path=self.audio_path, tags=self.tags, metadata=self._clip_metadata
        )


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO, sr=None) -> Tuple[np.ndarray, float]:
    """Load a  TAU Urban Acoustic Scenes 2019 audio file.

    Args:
        fhandle (str or file-like): File-like object or path to audio file
        sr (int or None): sample rate for loaded audio, None by default, which
            uses the file's original sample rate of 44100 without resampling.

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file

    """
    audio, sr = librosa.load(fhandle, sr=sr, mono=False)
    return audio, sr


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The  TAU Urban Acoustic Scenes 2019 dataset
    """

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="tau2019uas",
            clip_class=Clip,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
        )

    @core.copy_docs(load_audio)
    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)

    @core.cached_property
    def _metadata(self):
        metadata_path = os.path.join(
            self.data_home, "TAU-urban-acoustic-scenes-2019-development", "meta.csv"
        )

        splits = [
            "development.train",
            "development.evaluate",
            "development.test",
            "evaluation",
            "leaderboard",
        ]

        metadata_index = {}

        with open(metadata_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter="\t")
            next(csv_reader)
            for row in csv_reader:
                file_name = os.path.basename(row[0])
                clip_id = "{}/{}".format(
                    "development", os.path.basename(file_name).replace(".wav", "")
                )
                scene_label = row[1]
                identifier = row[2]
                city = identifier.split("-")[0]
                metadata_index[clip_id] = {
                    "scene_label": scene_label,
                    "city": city,
                    "identifier": identifier,
                }

        for split in splits:
            subset = split.split(".")[0]
            evaluation_setup_path = (
                "TAU-urban-acoustic-scenes-2019-{}/evaluation_setup".format(subset)
            )
            if subset == "development":
                fold = split.split(".")[1]
                evaluation_setup_file = os.path.join(
                    self.data_home, evaluation_setup_path, "fold1_{}.csv".format(fold)
                )
            else:
                evaluation_setup_file = os.path.join(
                    self.data_home, evaluation_setup_path, "test.csv"
                )

            with open(evaluation_setup_file) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter="\t")
                next(csv_reader)
                for row in csv_reader:
                    file_name = os.path.basename(row[0])
                    clip_id = "{}/{}".format(
                        subset, os.path.basename(file_name).replace(".wav", "")
                    )

                    if subset != "development":
                        metadata_index[clip_id] = {
                            "scene_label": None,
                            "city": None,
                            "identifier": None,
                        }

                    metadata_index[clip_id]["split"] = split

        return metadata_index
