#  PKSH: A Scalable High-Equalization Framework Leveraging Prior Knowledge for Intracranial Electric Field Focusing


This code is provided as a supplementary resource for an ICML paper with the reference number ***6920***.


-----------


## Environments
To set up the environment for running PKSH, use the command ```pip install -r requirements.txt```.

## Quick Start
Regarding the startup of PKSH, it is possible to start PKSHA with ```python TIS.py -t automti_I_T -t2 automti_3 -p motor -m hcpgroup -g 2 -en 75 -n PKSH -n2 PKSH -s 1 -ap yes``` or ```python TES.py -t autotdcs_I_T -t2 autotdcs_3 -p v1 -m hcpgroup -g 2 -en 75 -n PKSH -n2 PKSH -s 1 -mn 2 -ap yes```. All the parameters involved in PKSH and their descriptions are shown below.

| Parameter | Description                                       |
|-----------|---------------------------------------------------|
| `-t`      | Type of the first optimization                    |
| `-t2`     | Type of the second optimization                   |
| `-p`      | ROI                                               |
| `-m`      | Name of the human head model                      |
| `-g`      | Number of iterations                              |
| `-en`     | Number of candidate electrodes                    |
| `-n`      | Name of the first optimization result             |
| `-n2`     | Name of the second optimization result            |
| `-s`      | Random seed                                       |
| `-mn`     | Number of high-precision stimulation electrodes   |
| `-ap`     | Whether to stimulate all targets at the same time |


## Datasets
We will post our processing method and dataset later as the dataset is too large.
## Evaluation
You are able to evaluate the electric field focusing method inside the Evaluation folder.
