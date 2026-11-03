
# Setup for building ms.pdf

For the quarto JSS style, see https://github.com/quarto-journals/jss
```
quarto add quarto-journals/jss
```

The Python environment can be rebuilt with
```
make venv
source .venv/bin/activate
```
Or it can be built from scratch as follows:
```
cd ~/git/tutorials/article
python3.12 -m pip install virtualenv
rm -rf .venv
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip==25.2
pip install ~/git/pypomp
# install jax without cuda support
pip install -U jax
# asking for cuda support on a mac leads to download of
# an ancient jax version
# pip install -U "jax[cuda12]"

# for pypomp
pip install tqdm xarray pandas pytest

# for ms
pip install jupyter seaborn

```

