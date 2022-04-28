# Experiments for *"Spectral Bounds for Graph Echo State Network Stability"*

Accompanying code to reproduce experiments from the paper "Spectral Bounds for Graph Echo State Network Stability", accepted at The 2022 International Joint Conference on Neural Networks (IJCNN 2022).

### Run

An environment with [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) is required. To install the [Graph ESN library](https://github.com/dtortorella/graph-esn), simply do `pip install graphesn`.

To reproduce the graph classification experiments, run
```
python tu-graphs.py --dataset <dataset-name> --device <device> --units 16 32 64 128 256 512 --use rho --rho 0.1 0.2 0.4 0.6 0.8 1 1.2 1.4 1.6 1.8 2 2.2 2.4 2.6 2.8 3 3.2 3.4 3.6 3.8 4 4.2 4.4 4.6 4.8 5 5.2 5.4 5.6 5.8 6.0 6.2 6.4 --scale 1 --ld 1e0 1e-1 1e-2 1e-3 1e-4 1e-5 --trials 50 --batch-size 512
```

### Copyright

```
Copyright (C) 2022, Domenico Tortorella
Copyright (C) 2022, University of Pisa

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
```
