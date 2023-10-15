# HFT-AlphaAnalysis-Framework

This framework is specifically tailored for High-Frequency Trading (HFT) to study various combinations of factors (A) and targets (B). It provides a robust and efficient means to analyze minute-by-minute crypto data, leveraging different classify methods and factor calculation methods using handcrafted algorithms optimized for extensive datasets.

### Highlights
#### 1. Selective Factor Demos
For clarity and privacy, the framework selectively showcases factor demos, indicating the modularity and extensibility of the codebase for various factor analyses in HFT scenarios.

#### 2. Optimized for Large Datasets
This framework is ideally tailored for processing extensive datasets. As a demonstration, it has been tested on more than 15 feature datasets, each containing per-minute data for 195 cryptos spanning from February 1, 2022, to July 30, 2023. The provided demos highlight the perfect balance achieved between memory usage and execution time.

#### 3. Numpy-Based Algorithms
Prioritizing performance, all functions are meticulously handcrafted using only Numpy. This includes recreations of popular functions from mainstream libraries, thus ensuring lightning-fast execution speeds—a prerequisite in HFT. Pandas, in particular, is notoriously slow for such large-scale data, for example, `pandas.shift`, `pandas.rolling`, `pandas.apply`, `pandas.corrwith`, also include `sm.ols` from sklearn.

### Function Overview
The concept of factor slicing is rooted in three fundamental elements: the subject, the tool, and the outcome.
#### 1. Subject
This refers to the target variable, which should possess additivity. They can be re-combined or summed up.

#### 2. Tool
These are indicators capable of discerning information, serving as the core of the slicing theory. 

#### 3. Outcome
After slicing, further refinement of the information results in the final output. One can choose parts of the sliced data that are rich in information as a proxy for the new factor, essentially purifying the information.  This hidden step serves as an important "standardization" process. It provides a fair standardization level, by introduction some cross-sectional information, enhancing the factor's stability.

In essence, the philosophy behind the function is to segment data based on discerning indicators, refine this segmented data, and then use it for more insightful and standardized analyses.
