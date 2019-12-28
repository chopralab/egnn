# Energy Graph Neural Networks (EGNN)

This project contains the code and training/test data used for training and testing Energy Graph Neural Networks (EGNN). The code is a modified version of the Graph Neural Network code by masashitsubaki which is available [here](https://github.com/masashitsubaki/molecularGNN_smiles) under the Apache license.


## Requirements

### CANDOCK

A copy of the CANDOCK docking program is provided in the CANDOCK submodule directory. To build CANDOCK, you will need a modern **C++** compiler, the Boost Library, and CMake. Other requirements, such as *GSL* and *OpenMM* will be installed during the CMake build process. Note that CANDOCK is only required for reproducing the docking results shown in the associated paper. It is not required for running the associated *Python* scripts.

### EGNN

* PyTorch
* scikit-learn
* RDKit

## Usage

### Obtain the code

```bash
git clone https://github.com/chopralab/egnn.git
cd egnn
git submodule update --all
```

The above commands will download both the *EGNN* Python scripts required for training a model and the **C++** docking code for CANDOCK v0.6.0.

### Building CANDOCK

After doing the above, do the following where N is the number of processor cores on your machine:

```bash
cd candock
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j N
```

Note that the above commands are tested on Ubuntu 16.04 and 18.04.

### Running the EGNN code

All the files in the *input* are prepared for the compounds discussed in the paper. Each line of the text file corresponds to a compound used in the training (*data.txt*) and test sets (*test.txt*). THe first column is the SMILES string for the molecule, the next 96 columns are the 96 scores obtained from docking the compound with CANDOCK. In the **data.txt** file, there is an additional column to denote if the compound is active or inactive (with a 0 or 1, respectively).

After creating/editing the above two files, you must run the *preprocess_train_data.sh* script. If any changes to the compounds is made in either *data.txt* or *test.txt*, the script must be run again. You must supply the name of the dataset and the radius used for fingerprinting. By default, the dataset will be named PDL1 and the radius will be 2. This script will populate the folders *train* and *test* with the required data for running the model.

Once the training and test sets have been prepared, you can run the *train_full.sh* script to train the model where the weights will be placed in the *fullmodel* directory. You can edit the hyperparameters in the *train_full.sh* and you must ensure that the radius parameter matches that of the one used during preparation.

After the model finishes training, the *run_test.sh* script can be run, which will predict the activity of the compounds in the *test.txt* file. Note that hyperparameters must be the same as when the model was trained to ensure that correct weights are loaded. This script will automatically bootstrap the results and place the final results in the *bootstrapping_results* directory. The final results can be combined using the *combining_smiles_bootstrapping_average_countsover0.5.py* script.
