# Getting Started with Example Data
A step-by-step process to getting started with the AHINN package.
NB// Step 2 can be skipped without issue if you do not have access to the appropriate ERA5 data.

1. **Running `main.py` on Basic Settings**
   1. Get a single AHI scene folder of .DAT files and put it in the Example directory. If you do not already have access to AHI data, see https://www.eorc.jaxa.jp/ptree/registration_top.html.
   2. Run the `main.py` file without using ERA5 data or the default paths with `python main.py --use_defaults False --use_era False`.
   3. After sucessfully running, check there is a .nc file in the Example directory. This will contain the outputs for the example scene.
2. **Running `main.py` with ERA5 Data (Single and 37 Pressure Level monthly-averaged-by-hour)**
   1. If you have access to ERA5 monthly-averaged-by-hour data, assign the path to the directory which contains all the ERA5 data in defualts.json. This is done by changing the `path.ERADir` entry to the system-appropriate `"path/to/ERA/data"`. It is important to note that you should assign the directory to be as close to the ERA5 single and 37 pressure level files as possible, but the path must contain all the available data. The more specific you are with the path, the faster the code will run.
   2. Run the `main.py` file with ERA5 data and defaults activated with `python main.py --use_defaults True --use_era True`.
   3. Check in the Example directory for the .nc file. It should have `w-era` in the file name.
3. **Running from Defualt Directories**
   1.
   2.
