# Simula Summer Internship
## Analysis of Smart Home Dataset

A detailed report documenting the results of the internship can be found in  [documentation/Report.pdf](documentation/Report.pdf)

The dataset that was used in this repository is from [here](https://sites.google.com/site/tim0306/)
Kasteren, T.L.M. (2010).


To reproduce the results in the report, run the following commands

```bash
pip install -r requirements.txt

# Install entropy package
git clone https://github.com/raphaelvallat/entropy.git entropy/
cd entropy/
pip install -r requirements.txt
python setup.py develop

python main.py
```