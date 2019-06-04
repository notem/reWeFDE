If you use this project in your paper, please include the following as citations:

> Shuai Li, Huajun Guo, and Nicholas Hopper. 2018. Measuring Information Leakage in Website Fingerprinting Attacks and Defenses. In Proceedings of the 2018 ACM SIGSAC Conference on Computer and Communications Security (CCS '18). ACM, New York, NY, USA, 1977-1992. DOI: https://doi.org/10.1145/3243734.3243832

and

> Imani et. al. 2019. Mockingbird: Defending Against Deep-Learning-Based Website Fingerprinting Attacks with Adversarial Traces. arXiv:1902.06626

## Introduction

Purpose of this project is to reproduce the WeFDE information leakage results with code which is more user-friendly and expandable.
To achieve this, much of the original Matlab code has been replaced with analogous python code.
The application has also been designed to require minimal steps from the user.
Performing an website fingerprinting information leakage analysis should not require extensive understanding of the underlying analysis mechanisms.

All credit for the design of this system goes to Shuai et. al. [1].

## Project Overview

#### Requirements

* Base Matlab install (tested w/ 2018+)
* Python 2.70 & ``requirements.txt`` modules
* Compatible C++ compiler

#### Roadmap

* [Completed] Closed-world information leakage analysis
* Open-world analysis extension
* Bootstrapping results verification
* Re-write feature processing scripts to be extendable with additional features
* Replace Matlab @kde library with native Python implementation

#### Setup
Install required python modules:
```bash
pip install -r requirements.txt
```

Compile and install required Matlab @kde library:
```bash
matlab -nodisplay -nosplash -nodesktop -r "try, run('./info_leak/matlab/@kde/mex/makemex.m'), catch me, fprintf('%s / %s\n',me.identifier,me.message), end, exit"
```
#### Organization

###### Data processing
Before information leakage analysis can be performed on a dataset, that dataset must first be transformed to it's feature representation.
This process is handled by the scripts in the ``preprocess`` directory.
At present, this code is a revised variant of the processing scripts seen in the Shuai et. al. public code [2].

The main script processes all files in a nested directory.
Trace files should follow the format defined by Wang et. al. in [3].
Each website trace is saved as a CSV-type file using a space delimiter.
The name given to each transformed trace is the original trace name with ``.feature`` prepended.
These feature files will be loaded during the information leakage analysis.

###### Information leakage analysis


###### Results presentation
###### Further analysis

## Usage Examples

```bash
TRACE_PATH=/path/to/traces
FEATURE_PATH=/path/to/features
LEAKAGE_PATH=/path/to/leakage
```

###### Data processing

```bash
python preprocess/extract.py --traces ${TRACE_PATH} --output ${FEATURE_PATH}
```

###### Information leakage analysis

```bash
python info_leakage/info_leak.py --features ${FEATURE_PATH} --output ${LEAKAGE_PATH} \
                                 --n_samples 5000 --nmi_threshold 0.9 --topn 100 --n_procs 8
```

## References
[1] Shuai Li, Huajun Guo, and Nicholas Hopper. Measuring Information Leakage in Website Fingerprinting Attacks and Defenses. In ACM Conference on Computer and Communications Security, 2018.

[2] https://github.com/s0irrlor7m/InfoLeakWebsiteFingerprint

[3] Tao Wang. Website fingerprinting: attacks and defenses. PhD thesis, University of Waterloo, 2016.
