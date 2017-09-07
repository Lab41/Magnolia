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
  "metadata_file": "...", // location of csv metadata file used to partition data
  "output_directory": "...", // output directory where .txt are to be stored
  "description": "...", // textual description of partitions
  "rng_seed": null, // random number seed (for reproducibility)
  "partition_graphs": [ // (described later)
  ]
}
```

The `data_file` is the HDF5 file that contains the actual data to be
partitioned.
`metadata_file` is the location of the CSV file containing the metadata used for
partitioning the data.
The `output_directory` will contain the results of the partitioning in the form
of sub-directories and CSV files each of which contain a list of HDF5 `Dataset`s
that belong to that group.
The `description` is an optional plain text description of how the data is
partition.
This is useful for describing the intent for the `partition_graphs`.
The `partition_graphs` is a specially-formatted portion of the settings that
fully specifies how the input file is partitioned.
Finally, the `rng_seed` is an integer seed used to initialize the random number
generator.

### The partition graph

The partition graph is a graphical representation of how to partition a
data table where each entry in the table conceptually represents a unique data
sample.
It is a directed, acyclic graph where each node specifies how to filter the
table (i.e. a Pandas `query`) and the edges specify the proportion of the parent
node to pass along to the next node.
The proportion of the parent node to pass along an edge may be grouped along
categories in column of the table or over all samples.
The terminating nodes of the graph (or leaves on a tree) are the mutually
exclusive groups of data samples.
A generic table is given below that will aid in the following discussion.

![Generic_data table](images/generic_data_table.png)

Here, the number of samples is $N$ while the number of columns is $M$.
The column that contains the unique, identifying information regarding each
individual sample within an HDF5 file is highlight in green and denoted $L_K$.

Generally speaking, to partition a dataset, only a three operations are
possible:
* Filtering through queries
* Random splitting (on either categories or samples)
* Assign to groups the results of filtering and splitting
If a column contains categorical values, these columns could be split
proportionally to the number of samples within those categories.
Conceptually, the whole dataset is only ever filtered and split until what
remains is stored in groups.
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

![UrbanSound8K_file_hierarchy](images/UrbanSound8K_metadata_table.png)

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
The first node in the graph represents the filter on the salience category
(alluded to by the matching green coloring).
The next node filters the class category and the edge connecting the first and
second node indicates all the categories from the first filter should propagate
to the second node (there is only one category that passes the first node's
filter, thus, one can only ever pass 100% of this category to the next node).
The next two edges specify that 20% of the non-`children_playing` noise class
categories be reserved for the out-of-sample test group while the other 80% of
the categories will be sent through another filter.
This last filter passes all categories (data file names in this case) through
to the training, validation, and in-sample test groups via an 80/10/10 split.


#### How the split is determined

The split at each directory level is calculated by considering the total number
of data files in all it's sub-directories.
Thus, the proportion specified along each edge is only approximately followed
as categories are discrete.
Print outs of the specified and actual tree structures are given after the
partitioning is finished.

#### Partition graph JSON structure

The corresponding JSON to specify the UrbanSound8K partition graph previously
diagramed is given below.

```javascript
{
  "partition_graphs": [
    {
      "data_label": "key",
      "filters": [
        {
          "id": "time_volume_interference",
          "pandas_query": "salience == 1 & duration >= 2 & Class != \"children_playing\""
        },
        {
          "id": "training_set"
        }
      ],
      "groups": ["out_of_sample_test", "in_sample_test", "validation", "train"],
      "splits": [
        {
          "source": "time_volume_interference",
          "target": "out_of_sample_test",
          "split_on": "Class",
          "fraction": 0.1
        },
        {
          "source": "time_volume_interference",
          "target": "training_set",
          "split_on": "Class",
          "fraction": 0.9
        },
        {
          "source": "training_set",
          "target": "in_sample_test",
          "fraction": 0.1
        },
        {
          "source": "training_set",
          "target": "train",
          "fraction": 0.8
        },
        {
          "source": "training_set",
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
