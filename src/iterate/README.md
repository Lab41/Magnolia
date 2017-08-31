# Data Iteration

This folder contains the algorithms responsible for partitioning the data,
iterating over those partitions, and mixing different sources of audio.

## Data Partitioning

Partitioning of a preprocessed data file into various sets is accomplished
through the `partition_data.py` script.
This script takes as arguments a preprocessing settings file, logging
configuration file, and logger name.
Argument names can be displayed by running the script with the `--help` or `-h`
flag.
The most pertinent argument is the partition settings JSON file; a template
of which can be found in `magnolia/settings/partition_template.json`.
The structure of the JSON file is as follows:

```javascript
{
  "data_file": "...", // HDF5 file to partition
  "output_directory": "...", // output directory where .txt are to be stored
  "description": "...", // textual description of partitions
  "partition_graphs": [ // (described later)
  ]
}
```

The `output_directory` will contain the results of the partitioning in the form
of text files each of which contain a list of HDF5 `Dataset`s that belong to
that group.
The `description` is an optional plain text description of how the data is
partition.
This is useful for describing the intent for the `partition_graphs`.
The `partition_graphs` is a specially-formatted portion of the settings that
fully specifies how the input file is partitioned.

### The partition graph

The partition graph is a graphical representation of how to partition a
hierarchically structured file system (HDF5 in this case).
It is a directed, acyclic graph where each node specifies how to filter the
directories (referred to as categories onward) at a directory level and the
edges specify the proportion of the node categories or files to keep.
For reference, a generic hierarchical file system directory structure is given
below.

![Generic_file_hierarchy](images/file_hierarchy.png)

Here, each directory level (colored green and red and denoted by $A$ and $B$)
contains an arbitrary number of categories (or HDF5 groups) denoted
$\{A_1\ldots A_n\}$ and $\{B_1\ldots B_k\}$.
The data files are located at any directory level and are denoted by the yellow
icons at the bottom of the diagram.
To group the data files stored in this hierarchy, one could imagine selecting
some subset of categories within the first directory level ($A$).
This subset selection will be called "filtering."
Once the first level of categories is filtered, one can then ask that the
remaining categories be proportionally split.
After this split, one can repeat the filtering process at the next directory
level and split those categories if desired.
Finally, at the level(s) of the individual data files, these can again be
filtered and split and the remaining data file names be stored in a text file.
This is abstractly shown in the following diagram.

![Generic_file_hierarchy_partition](images/file_hierarchy_filter_split.png)

Each node here represents a filter ($F$) at a certain directory level while the
edges are the fractions of category labels being passed along.
The final data file names are stored in one or more groups ($G$) and saved to
text files.
It's important to note that the fractions associated with the edges emanating
from a node can sum to any number less than or equal to 1.

To make this more concrete, the following is the HDF5 file hierarchy for the
UrbanSound8K.

![UrbanSound8K_file_hierarchy](images/UrbanSound8K_file_hierarchy.png)

Here, the first directory level is the salience and the second level is the
class of noise.
The data files are then all contained within the class categories.
Suppose the following partitioning is desired: Select a salience of 1, remove
the "children_playing" noise class, reserve 20% of the remaining noise classes
for out-of-sample testing, and, finally, make an 80/10/10 train/validation/test
split of the remaining 80%.
This is what is diagramed in the following partition graph.

![UrbanSound8K_file_hierarchy_partition](images/UrbanSound8K_file_hierarchy_filter_split.png)

Here, the two test groups have different names, but in general they could've
been named the same for a single test group.

#### How to determine the split?

The split at each directory level is calculated by considering the total number
of data files in all it's sub-directories.
Thus, the proportion specified along each edge is only approximately followed
as categories are discrete.
A print out of the specified and actual tree structures are given after the
partitioning is finished.

#### Partition graph JSON structure

The corresponding JSON to specify the UrbanSound8K partition graph previously
diagramed is given below.

```javascript
{
  "partition_graphs": [
    {
      "filters": [
        {
          "id": "salience=1",
          "only": ["1"]
        },
        {
          "id": "class!=children_playing",
          "except": ["children_playing"]
        },
        {
          "id": "all files",
        }
      ],
      "groups": [
        {
          "name": "out_of_sample_test"
        },
        {
          "name": "in_sample_test"
        },
        {
          "name": "validation"
        },
        {
          "name": "train"
        }
      ],
      "splits": [
        {
          "source": "salience=1",
          "target": "class!=children_playing",
          "fraction": 1.0
        },
        {
          "source": "class!=children_playing",
          "target": "all files",
          "fraction": 0.8
        },
        {
          "source": "class!=children_playing",
          "target": "out_of_sample_test",
          "fraction": 0.2
        },
        {
          "source": "all files",
          "target": "train",
          "fraction": 0.8
        },
        {
          "source": "all files",
          "target": "in_sample_test",
          "fraction": 0.1
        },
        {
          "source": "all files",
          "target": "validation",
          "fraction": 0.1
        }
      ]
    }
  ]
}
```

Each `filter` can have the following attributes (default is to include all
categories):
* `id`: name of the filter that is referenced by the splits (required)
* `only`: only include the following matched categories or data files (list or regex,
  optional)
* `except`: exclude the following match categories (list or regex, optional)

Each `group` has the same attributes as the `filter`s except that `id` is
replaced by `name`.

Each `split` has the following attributes:
* `source`: `id` of a `filter`
* `target`: `id` of a `filter` or `group` (at a directory level deeper than
  `source`)
* `fraction`: split fraction of categories or data files

## Iteration

## Mixing
