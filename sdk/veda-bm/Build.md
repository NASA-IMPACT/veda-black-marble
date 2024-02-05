### Build instructions

* Increase the build version in setup.cfg
```
python3 -m venv venv
source venv/bin/activate

rm -rf dist
python3 -m pip install --upgrade build
python3 -m build
python3 -m pip install --upgrade twine
python3 -m twine upload --repository pypi dist/*
```