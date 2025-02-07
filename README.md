# Efficient Radiation Treatment Planning based on Voxel Importance

This repository contains the source code of the paper [**Efficient Radiation Treatment Planning based on Voxel Importance**](https://iopscience.iop.org/article/10.1088/1361-6560/ad68bd) which is published in the journal [Physics in Medicine & Biology](https://iopscience.iop.org/journal/0031-9155).

## Abstract
*Objective.*
Radiation treatment planning involves optimization over a large number of voxels, many of which carry limited information about the clinical problem.
We propose an approach to reduce the large optimization problem by only using a representative subset of informative voxels.
This way, we drastically improve planning efficiency while maintaining the plan quality.

*Approach.*
Within an initial probing step, we pre-solve an easier optimization problem involving a simplified objective from which we derive an importance score per voxel.
This importance score is then turned into a sampling distribution, which allows us to subsample a small set of informative voxels using importance sampling.
By solving a -- now reduced -- version of the original optimization problem using this subset, we effectively reduce the problem's size and computational demands while accounting for regions where satisfactory dose deliveries are challenging.

*Main results.*
In contrast to other stochastic (sub-)sampling methods, our technique only requires a single probing and sampling step to define a reduced optimization problem.
This problem can be efficiently solved using established solvers without the need of modifying or adapting them.
Empirical experiments on open benchmark data highlight substantially reduced optimization times, up to 50 times faster than the original ones, for intensity-modulated radiation therapy (IMRT), all while upholding plan quality comparable to traditional methods.

*Significance.*
Our novel approach has the potential to significantly accelerate radiation treatment planning by addressing its inherent computational challenges.
We reduce the treatment planning time by reducing the size of the optimization problem rather than modifying and improving the optimization method.
Our efforts are thus complementary to many previous developments.

## Code

The code was tested with the following versions:

* python 3.9.0
* numpy 1.23.2
* scipy 1.9.0
* cvxpy 1.4.1
* fire 0.4.0
* matplotlib 3.5.3
* [mosek](https://www.mosek.com/) 10.1

## Data

For our experiments, we use the CORT dataset:

*Craft D, Bangert M, Long T, Papp D and Unkelbach J 2014 [Shared data for intensity modulated radiation therapy (IMRT) optimization research: the CORT dataset](https://academic.oup.com/gigascience/article/3/1/2047-217X-3-37/2682969) GigaScience 3 2047–217X*


### Files

- `CORT.py` contains code to load the files of the CORT data set.
- `config.py` contains settings such as the gantry and couch angles, structures, doses, and optimization hyperparameters about the cases.
- `plot_DVHdist.py` generates Figures 2, B3, and B4.
- `plot_DVHerror_time.py` generates Figures 3 (right), 4, B5, B6, and B7.
- `plot_PGD_DVH.py` generates Figures 3 (left) and B2.
- `plot_dist_samples.py` generates Figures 1 and B1.
- `run_experiment.py` is used to run experiments. It will create result files of the following form `result_{case}_{score_method}_{m}_{repetitions}_{w_BDY_over}_{w_OAR_over}_{w_PTV_over}_{w_PTV_under}.npz`
- `test_CORT.py` contains the tests given in the CORT paper to verify that the data loading is correctly handled.
- `utils.py` contains the implementation of plotting dose volume histograms, computing radiation treatment plans, computing the objective function, computing subsets, and computing scores.


## Running Experiments

To run all experiments, one would have to execute:
``` bash
# Prostate case
for loss in squared; do
  python run_experiment.py --case=Prostate --loss=$loss --score_method=full --m=0 --repetitions=0 --w_BDY_over=1.0 --w_OAR_over=1.0 --w_PTV_over=4096.0 --w_PTV_under=4096.0;

  # run the following percentages: 0.1 0.25 0.5 0.75 1 2.5 5 7.5 10 20 25 30 40 50 60 70 75 80 90
  for m in 690 1726 3452 5178 6904 17259 34519 51778 69037 138075 172593 207112 276149 345186 414224 483261 517780 552298 621336; do
    for score_method in uniform gradnorm; do
      python run_experiment.py --case=Prostate --loss=$loss --score_method=$score_method --m=$m --repetitions=50 --w_BDY_over=1.0 --w_OAR_over=1.0 --w_PTV_over=4096.0 --w_PTV_under=4096.0;
    done;
  done;
done

# Liver case
for loss in squared; do
  python run_experiment.py --case=Liver --loss=$loss --score_method=full --m=0 --repetitions=0 --w_BDY_over=1.0 --w_OAR_over=1.0 --w_PTV_over=4096.0 --w_PTV_under=4096.0;

  # run the following percentages: 0.1 0.25 0.5 0.75 1 2.5 5 7.5 10 20 25 30 40 50 60 70 75 80 90
  for m in 1927 4818 9637 14455 19274 48184 96368 144552 192736 385471 481839 578207 770943 963678 1156414 1349150 1445518 1541886 1734621; do
    for score_method in uniform gradnorm; do
      python run_experiment.py --case=Liver --loss=$loss --score_method=$score_method --m=$m --repetitions=50 --w_BDY_over=1.0 --w_OAR_over=1.0 --w_PTV_over=4096.0 --w_PTV_under=4096.0;
    done;
  done;
done

# HeadAndNeck case
for loss in squared; do
  python run_experiment.py --case=HeadAndNeck --loss=$loss --score_method=full --m=0 --repetitions=0 --w_BDY_over=1.0 --w_OAR_over=1.0 --w_PTV_over=4096.0 --w_PTV_under=4096.0;

  # run the following percentages: 0.1 0.25 0.5 0.75 1 2.5 5 7.5 10 20 25 30 40 50 60 70 75 80 90 
  for m in 252 630 1260 1889 2519 6297 12595 18892 25189 50379 62973 75568 100757 125946 151136 176325 188920 201514 226704; do
    for score_method in uniform gradnorm; do
      python run_experiment.py --case=HeadAndNeck --loss=$loss --score_method=$score_method --m=$m --repetitions=50 --w_BDY_over=1.0 --w_OAR_over=1.0 --w_PTV_over=4096.0 --w_PTV_under=4096.0;
    done;
  done;
done
```
This can be split to multiple nodes. Every run of `run_experiment.py` will create a result file of the following form: `result_{case}_{score_method}_{m}_{repetitions}_{w_BDY_over}_{w_OAR_over}_{w_PTV_over}_{w_PTV_under}.npz`.

