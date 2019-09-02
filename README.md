# ClimTools

ClimTools is a library of tools thought for the analysis and visualization of climate datasets.

It is constituted by two modules:
1) **climtools_lib.py** -> a miscellaneous library of basic functions that perform the following operations:
- reading and saving of netcdf files (.nc), either using the python netCDF4 package or the python iris package;
- basic computations: daily/monthly anomalies, trends, seasonal statistics, ...;
- EOF calculation using the python eofs module;
- a set of functions used for the calculation of Weather Regimes (see WRtool -> https://github.com/fedef17/WRtool): clustering (using Kmeans), operations on clusters, projections/rotations in the EOF space, ...;
- visualization: single/multiplot contour maps, Taylor plot, animations..

2) **climdiags.py** -> a higher level library that contains some more complex tools. Currently the following tools are included:
- WRtool : a tool for the computation of Weather Regimes and related statistics, see WRtool -> https://github.com/fedef17/WRtool;
- heat_flux_calc : a tool for the computation of meridional heat fluxes starting from 3D high frequency wind, humidity and temperature fields.

## How to install and use ClimTools

### Preliminaries
The following steps are needed to be able to use the ClimTools library. Perform them all the first time you're using it, it won't take so long!

#### -1. Clone the ClimTools repo
Open a terminal inside your code folder and do:
```
git clone https://github.com/fedef17/ClimTools.git
```

This will create a folder ClimTools inside your code folder.
To update your local version of the code, just go inside the ClimTools folder and do:
```
git pull origin master
```

#### 0. Install Anaconda
ClimTools needs a Conda environment, which is defined by the file **env_ctl3.yml**. If you don't have Conda installed, follow the instructions at https://conda.io/projects/conda/en/latest/user-guide/install/index.html?highlight=conda.

#### 1. Create the *ctl3* environment
To create the environment, open a terminal from your ClimTools folder and do:
```
conda env create -f env_ctl3.yml
```
Each time you want to use ClimTools, remember to activate the environment first:
```
conda activate ctl3
```

#### 2. Compile the Fortran routines
When the conda environment is correctly installed, you can proceed with the following.
Some functions inside climtools_lib.py require the use of Fortran routines which can be found inside the cluster_fortran/ folder. To make them work, open a terminal from your ClimTools folder and do:
```
cd cluster_fortran
chmod +x compile_py3.sh
./compile_py3.sh
```

In alternative, you might just open the file compile_py3.sh and copy and run each command in shell.
This operation will create two binary libraries *ctool.so* and *ctp.so* in your ClimTools folder.

#### 3. Add permanently the path to ClimTools to your PYTHONPATH
Open a terminal inside the ClimTools folder and activate the ctl3 environment (*conda activate ctl3*).

Then do:
```
conda-develop .
```
This commands adds the path to the ClimTools folder to the *ctl3* environment system paths.

### Usage

ClimTools gives you two Python modules. To use the functions inside them, just put at the top of your code:
```
import climtools_lib as ctl
import climdiags as cd
```

And then you can play around with the functions.. For example, to produce a simple contour map from a netcdf file:
```
from matplotlib import pyplot as plt

var, coords, aux_info = ctl.read_iris_nc('filename.nc')
map = ctl.plot_map_contour(var, coords['lat'], coords['lon'])

plt.show()
```
