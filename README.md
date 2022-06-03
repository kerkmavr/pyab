# pyab
An experimentation library in order to conduct analysis after an experiment.
There is no randomisation/splitting logic code included but only the analysis part considering: 
   "fixed horizon" experiments which follow the frequentist approach and
   "bayesian" experiments. 
In regard to metrics, proportions/ratio metrics and cumulative figures are taken into account.
Proportions include metrics such as "pick-up ratio" and cumulative have to do with any revenue metric (not unitary). Normally the first category has to do predominantly with "Marketplace Ops" while the latter touches mostly "Growth" experiments. 

# Development Environment Setup
Below are some instructions on how to set up your development environment using `pyenv` and `pyenv-virtualenv`. You are of course free to use any virtual environment you prefer. More details about pyenv can be found [here](https://realpython.com/intro-to-pyenv).
1. Install `pyenv` and `pyenv-virtualenv` using brew:

   `brew install pyenv pyenv-virtualenv` (you may need to first install [homebrew](https://brew.sh/)).
2. Add the following lines to `.bashrc` / `.zshrc`:

   `eval "$(pyenv init -)"`

   `eval "$(pyenv virtualenv-init -)"`.
3. Restart your terminal.
4. Clone or fork the repository.
5. Go into the project directory: `cd pyab`.
6. Install the desired Python version:

   `pyenv install 3.8.13`.
7. Create a virtual environment with the desired Python version:

   `pyenv virtualenv 3.8.13 pyab_env`.
7. Activate the environment: `pyenv local pyab_env`.
8. Install the necessary requirements: `pip install -r requirements.txt`.
