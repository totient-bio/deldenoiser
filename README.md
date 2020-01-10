# deldenoiser

*Command line tool to remove effects of truncated side-products from read count data of a DNA-encoded library (DEL) screen.*

Table of Contents

* [Summary](#summary)
* [Installation](#installation)
* [Usage](#usage)
  * [Inputs](#inputs)
  * [Outputs](#outputs)
* [Documentation](#documentation)

<a name="summary"></a>
## Summary 

Sequencing read counts from a DEL screen are used as input.
The main output is the list of *fitness coefficients* for the compounds. Fore ach compound, they are proportional to its surviving fraction during binding assay. The following analysis steps are carried out by deldenoiser command line tool:

1. Estimate **tag imbalance** from pre-selection read counts. (Only if such data is available.)

2. Estimate **fitness of truncated compounds** using post-selection read counts, yields and tag imbalances factors.

3. Estimate **fitness of full-cycle compounds** using fitness of truncates.

4. Computate **breakdown of read counts** by truncation pattern.

It is assumed that yields of synthesis reactions are known, and the true fitness vector is sparse i.e. only a small minority of the DEL compounds have significant binding strength.

Note: We use a micro-fluidics-inspired terminology and refer to the different reactions that are run in parallel in each synthesis cycle as "lanes".

<a name="installation"></a>
## Installation

**Option 1:** Install to local python environment (requires Python 3.6 or higher) from pypi by running

```
pip install deldenoiser
```

**Option 2:** Install to local python environment from github by running 

```
git clone https://github.com/totient-bio/deldenoiser.git
pip install -e ./deldenoiser
```

**Option 3:** Build a local docker image `deldenoiser:<commit_hash>` by running 

```
git clone https://github.com/totient-bio/deldenoiser.git
cd deldenoiser
make docker_image
```

<a name="usage"></a>
## Usage

For a complete example, see [example/run_deldenoiser\_command\_line\_tool.bash](example/run_deldenoiser_command_line_tool.bash), which reads input files from [example/input/](example/input/) and writes results to [example/output/](example/output/).

Generally, running the command

```
deldenoiser --design <DEL_design.tsv>
            --postselection_readcount <readcounts_post.tsv>
            --output_prefix <prefix>
            [--dispersion <dispersion>]
            [--regularization_strength <alpha>]
            [--yields <yields.tsv>]
            [--preselection_readcount <readcounts_pre.tsv>]
            [--maxiter <maxiter>]
            [--tolerance <tol>]
            [--parallel_processes <processes>]
            
```
produces 5 files,

* `<prefix>_fitness.tsv`
* `<prefix>_truncate_fitness.tsv`
* `<prefix>_count_breakdown.tsv`
* `<prefix>_tag_imbalance.tsv`
* `<prefix>_inventory.tsv`

<a name="inputs"></a>
### Inputs

1. `<DEL_design.tsv>`, file of tab-separated values that encode the number of lanes in each cycle. It has two columns:
    * `cycle`: cycle index (1,2, ... cmax)    
    * `lanes`: number of lanes in the corresponding cycle (must be >= 1)

2. `<readcounts_post.tsv>`, file of tab-separated values that encode the read counts obtained from sequencing done *after* the DEL selection steps, with cmax + 1 columns:
    * `cycle_1_lane`: lane index of cycle 1
    * `cycle_2_lane`: lane index of cycle 2
    * ...
    * `cycle_<cmax>_lane`: lane index of cycle cmax
    * `readcount`: number of reads (non-negative integers) of DNA tags corresponding to each lane index combination.

3. `<prefix>`, string used to give unique name the output files, (it can also contain a path).
 
**Optional inputs:**

4. `<dispersion>`, dispersion parameter for the dispersed Poisson noise, (optional, default: 1.0)

5. `<alpha>`, regularization strength parameter, the higher the more sparse solutions are favored (optional, default: 1.0)
 
4. `<yields.tsv>`, file of tab-separated values that encode yields of each reaction of synthesis, with three columns (optional, default: all yields are set to 0.5):
    * `cycle`: cycle index (1,2, ... cmax)
    * `lane`: lane index (1,2, ... [max lane index in cycle])
    * `yield`: yield of reaction in the corresponding lane (0.0 .. 1.0)

6. `<readcounts_pre.tsv>`, file with the same structure as `<readcounts_post.tsv>`, but for reads obtained from sequencing done *before* the DEL selection step, (optional, default: assumed to be uniform across all sequences.)

7. `<maxiter>`: maximum number of coordinate descent iterations during fitting truncate fitness coefficients (default: 20)

8. `<tol>`: tolerance, if the total Poisson intensity of truncates changes less than this between consecutive iterations of coordinate descent, then fitting is stopped even before the number of iterations reaches maxiter (optional, default: 0.1)

9. `<processes>`: maximum number of parallel processes deldenoiser is allowed to start during fitting truncates (optional, default: number of system CPUs)


<a name="outputs"></a>
### Outputs

1. `<prefix>_fitness.tsv`, tab-separated values with the same index columns as the readcount input files, and three columns containing the mode, mean and min. stdev of the estimated fitness coefficients of the corresponding full-cycle products.
    * `cycle_<cid>_lane`: lane index of cycle cid = 1,2,... cmax
    * `fitness_mode`: estimated fitness of full-cycle compounds (mode of the posterior)
    * `fitness_mean`: estimated fitness of full-cycle compounds (mean of the posterior)
    * `fitness_minimum_stdev`: estimated lower bound on the posterior standard deviation of the fitness of full-cycle compounds

2. `<prefix>_truncate_fitness.tsv`, tab-separated values encoding the fitness coefficients of the truncates, each identified by their *extended lane index*. The cmax + 1 columns contain
    * `cycle_<cid>_lane`: extended lane index (which can take 0 as well, marking synthesis cycles that failed) of cycle cid = 1,2,... cmax
    * `fitness`: estimated fitness coefficients truncated compounds

3. `<prefix>_count_breakdown.tsv`, tab-separated values with the same index columns as the read count input files, and one columns containing the breakdown of the read counts by success pattern:
    * `cycle_<cid>_lane`: lane index of cycle cid = 1,2,... cmax
    * `readcount_<pattern>`: estimated fractional read count associated with each success pattern. Success pattern is a string of cmax number of "0" and "1" characters. (E.g. if cmax = 3, one of the columns will be `readcounts_010`, which will contain the read counts of the truncates associated with the failure of both the 1st and the 3rd cycle.) Note: The row sums are equal to counts from `<readcounts_post.tsv>` (up to floating point precision).

4. `<prefix>_tag_imbalance.tsv`, tab-separated values with the index columns as the read count input files, and a column of tag imbalances that estimated from pre-selection data (If no pre-selection data is provided, this file is still created, but it contains uniform values):
    *  `cycle_<cid>_lane`: lane index of cycle cid = 1,2,... cmax
    *  `tag_imbalance`: estimated imbalance of tags due to difference in nucleotide content, normalized to have mean of 1.

5. `<prefix>_inventory.tsv`, tab-separated values with the same index columns as the read count input files, but instead of the read counts this file contains the fractional composition associated with each success pattern. It is directly computed from the yields.
    * `cycle_<cid>_lane`: lane index of cycle cid = 1,2,... cmax
    * `fraction_of_<pattern>`: overall yield of the compound corresponding the success pattern, which is a string of cmax number of "0" or "1" characters. (E.g. if cmax = 3, one of the columns is `fraction_of__010`, which contains the fractional amount of the truncates due the failure of both the 1st and the 3rd cycle.)



<a name="documentation"></a>
## Documentation

* API documentation of deldenoiser Python package can be built by cloning the repository and running ```make docs``` command from the main directory, containing the Makefile.

* Developer's notes can be found at [development-notes/deldenoiser-development-notes.pdf](development-notes/deldenoiser-development-notes.pdf)

