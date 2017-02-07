<a href="http://ascl.net/1507.005"><img align="right" src="https://img.shields.io/badge/ascl-1507.005-blue.svg?colorB=262255" alt="ascl:1507.005" /></a>
<a href="LICENSE"><img align="right" hspace="3" alt="Code distributed under the open-source MIT license" src="http://img.shields.io/:license-mit-blue.svg"></a>

# slimplectic

`slimplectic` is a `python` implementation of the discrete non-conservative numerical integrator
of Tsang, Galley, Stein, and Turner (2015) [[arXiv:1506.08443]](http://arxiv.org/abs/1506.08443). It is based
on applying a discrete variational integrator approach
(e.g. [Marsden and West (2001)](http://lagrange.mechse.illinois.edu/pubs/MaWe2001/))
to our earlier paper on a stationary action principle for non-conservative dynamics
([Galley, Tsang, and Stein (2014)](http://arxiv.org/abs/1412.3082)).

## Quick start

Coming soon... for now, why not try following along with one of the
[three](Damped_Oscillator_SlimplecticGGLvsRK.ipynb)
[example](Poynting-Robertson_Cartesian-Long.ipynb)
[notebooks](PostNewtonian_Inspiral_with_RK.ipynb)?

![Screen shot of `slimplectic` `ipython` notebook](/../screenshots/screen1.png)

## Dependencies

`slimplectic` relies on fairly standard packages:
* [`numpy`](http://www.numpy.org/)
* [`scipy`](http://scipy.org/)
* [`sympy`](http://www.sympy.org/)

The example notebooks also require
* [`ipython`](http://ipython.org/)
* [`matplotlib`](http://matplotlib.org/)
