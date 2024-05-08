# Quantum Computing-Enhanced Algorithm Unveils Novel Inhibitors for KRAS

This repository contains code used to reproduce the results presented in the article "Quantum Computing-Enhanced Algorithm Unveils Novel Inhibitors for KRAS". The paper is co-authored by a diverse team including Mohammad Ghazi Vakili, Christoph Gorgulla, AkshatKumar Nigam, Dmitry Bezrukov, Daniel Varoli, Alex Aliper, Daniil Polykovsky, Krishna M. Padmanabha Das, Jamie Snider, Anna Lyakisheva, Ardalan Hosseini Mansob, Zhong Yao, Lela Bitar, Eugene Radchenko, Xiao Ding, Jinxin Liu, Fanye Meng, Feng Ren, Yudong Cao, Igor Stagljar, Alán Aspuru-Guzik, and Alex Zhavoronkov.

## Abstract

The discovery of small molecules with therapeutic potential is a long-standing challenge in chemistry and biology. Researchers have increasingly leveraged novel computational techniques to streamline the drug development process to increase hit rates and reduce the costs associated with bringing a drug to market. To this end, we introduce a quantum-classical generative model that seamlessly integrates the computational power of quantum algorithms trained on a 16-qubit IBM quantum computer with the established reliability of classical methods for designing small molecules. Our hybrid generative model was applied to designing new KRAS inhibitors, a crucial target in cancer therapy. We synthesized 15 promising molecules during our investigation and subjected them to experimental testing to assess their ability to engage with the target. Notably, among these candidates, two molecules, ISM061-018-2 and ISM061-22, each featuring unique scaffolds, stood out by demonstrating effective engagement with KRAS. ISM061-018-2 was identified as a broad-spectrum KRAS inhibitor, exhibiting a binding affinity to KRAS-G12D at 1.4μM. Concurrently, ISM061-22 exhibited specific mutant selectivity, displaying heightened activity against KRAS G12R and Q61H mutants. To our knowledge, this work shows for the first time the use of a quantum-generative model to yield experimentally confirmed biological hits, showcasing the practical potential of quantum-assisted drug discovery to produce viable therapeutics. Moreover, our findings reveal that the efficacy of distribution learning correlates with the number of qubits utilized, underlining the scalability potential of quantum computing resources. Overall, we anticipate our results to be a stepping stone towards developing more advanced quantum generative models in drug discovery.

## Article Link

- [Read the full article on arXiv](https://doi.org/10.48550/arXiv.2402.08210)

## Repository Content

- Code files to reproduce the computational experiments described in the paper.
- Additional resources and supplementary information related to the research.

### Supplementary Material

The supplementary material for this project can be found here:

- [Supplementary Information PDF](docs/si/SI_v0.pdf)

## Requirements

This repository's code is written in Python, utilizing various packages such as `Orquestra`, `PyTorch`, `SYBA`, `SELFIES`, and `Qiskit`. To run the code, you need to install these packages.

## Installation

To set up your environment to use this repository, install the following packages:

```bash
pip install orquestra-qml-core
pip install torch
pip install syba
pip install selfies
pip install qiskit
```

## How to Use
### Configuration

Before running the simulations, edit the configuration file benchmark_models_settings_qcbm.json to set up the parameters according to your experimental setup. This file contains settings such as model parameters, computation settings, and other options that affect the performance and outcome of the simulations.

### Submitting Jobs to SLURM

After configuring the settings, submit the job to the SLURM cluster by running the following command:

```bash 
python benchmark_models_v0.py --config benchmark_models_settings_qcbm.json <prior_size>
```

Replace `<prior_size>` with the desired value according to your specific requirements. This parameter needs to be defined to execute the script properly.

### SLURM Submission

To submit this as a job on a SLURM cluster, you may need to create a SLURM script. Here’s an example script:

```bash 
#!/bin/bash
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2000
#SBATCH --job-name=16_pr_be
#SBATCH --output=logs/benchmark_%j.txt
#SBATCH --error=logs/benchmark_%j.err


module load gcccore gompi/2022a python/3.10.4 pytorch syba
python benchmark_models_v0.py --config benchmark_models_settings_qcbm.json 16

```

Adjust the `#SBATCH` directives to match the resources available in your SLURM cluster and the requirements of your project. Be sure to replace

## Contact

For more information, please contact Mohammad Ghazi Vakili:

- **Website:** [ghazivakili.com](http://ghazivakili.com)
- **Email:** [m.ghazivakili@utoronto.com](mailto:m.ghazivakili@utoronto.com)



## Contributions

Contributions to this repository are welcome. If you are interested in contributing, please read the `CONTRIBUTING.md` file for guidelines on how to get involved.



## License

This project is open-sourced under the MIT license. See the LICENSE file for more details.
