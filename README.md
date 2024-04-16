# motorcycle_mpc
motorcycle control by model predictive control

https://github.com/dlab-ut/motorcycle_mpc/assets/38370926/d8160356-b229-4a43-9383-0597d36d3659


## installation
Recommended to run under virtual environment (venv or docker)

### venv ver
```
git clone https://github.com/dlab-ut/motorcycle_mpc
cd motorcycle_mpc

python3 -m venv .venv
source .venv/bin/activate
pip3 install -e .[dev]
```

### docker ver (TODO)



## usage
```
# demo 
python3 src/pathTrack/pathTrack.py course1 -g
```
- [course1, course2]
- gui

## simulation config
[config.py](src/pathTrack/config/config.py)

## reference
- 3d model [Autonomous bicycle](https://autonomous-bicycle.readthedocs.io/en/latest/) (Apache License 2.0)
- cubic_spline_planner [PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics) (MIT License)

## Aknowledge
This work was supported by JSPS KAKENHI Grant Number 23K03896.
