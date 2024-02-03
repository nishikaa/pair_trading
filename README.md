# UCB MIDS Capstone - Pair Trading 

## Instructions for Poetry:

Instructions on installing the [python poetry](https://python-poetry.org/) package. You will need to install this tool on your machine by [following the instructions](https://python-poetry.org/docs/#installation). Make sure to **follow all instructions, including adding poetry to your PATH**. It's best not to use other `python` management systems simultaneously (i.e., do not activate an anaconda environment and then create a poetry environment within the other).

After installing, navigate to "pair-trading-foundations" folder and run:

```
poetry install
```

then 

```
poetry shell
```

Which will set-up the virtual environment for you based on the packages specified in the pyproject.toml file.

To add packages, do

```
poetry add package_name
```

then run 

```
poetry install
```

To update the package lists. Doing so will modify the poetry.lock and pyproject.toml files, which you should commit and push.