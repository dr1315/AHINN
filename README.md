# AHINN
A package for analysing AHI data using neural networks. These networks can retrieve a cloud mask, cloud phase and cloud top heights.

## Package structure:
* ***main.py*** - Wrapper for processing an AHI scene using all the models available in the Models directory for the given input, e.g. if --use_era True, will use `<nn_name>_w-era.h5` files. Will analyse a given scene (data stored as a folder of L1b .DAT files) and return the model outputs as .nc file in a specified directory. For more information, run `python main.py --help`.
* ***defaults.json*** - A python dictionary of the deafult paths to key directories. These can be set to make processing easier. If left `BLANK`, the `main.py` code will not use them and will require a full\path\to\data:
  * AHI base directory - The directory where AHI folders of .DAT files can be found.
  * ERA5 base diretory - The directory where ERA5 monthly-average-by-hour data can be found (both 37 pressure level and single level).
  * Save directory - The directory where output .nc files should be stored by default.
* ***Processing*** - Directory containing the files need to pre-process, analyse and post-process the AHI scene and model outputs:
  * ***preprocessor.py*** - Contains all the code necessary to preprocess an AHI scene and model training data.
  * ***postprocessor.py*** - Contains all the code necessary to postprocess the model outputs and save them in a .nc file.
  * ***kerasNN.py*** - Contains all the code related to loading, proccesing and evaluting neural networks.
  * ***plotter.py*** - Contains all the code need to plot the outputs from a .nc file of model outputs, as well as code to generate RBGs and heatmaps of an AHI scene
* ***Model*** - Directory containing all the models that can be used for analyse. Also conatins a `optimal_thresholds.json` file which holds the optimal thresholds to be applied to binary output models.
* ***Example*** - Read through `GettingStarted.md` in this directory for help with ensuring the AHINN package is working correctly.
