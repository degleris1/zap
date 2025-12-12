.. zap documentation master file, created by
   sphinx-quickstart on Mon Nov 25 13:42:49 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. Add your content using ``reStructuredText`` syntax. See the
   `reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_
   documentation for details.

zap documentation
=================


**Zap**  is a Python library for for differentiable network optimization.
Currently, it provides a modular, component-based approach for modeling electricity systems and congestion-controlled flow networks.
Models can be solved using a GPU-accelerated ADMM solver, integrated into PyTorch-based deep learning models, or iteratively planned using gradient algorithms.

.. note::

   This project is under active development.


.. toctree::
   :hidden:
   :maxdepth: 1

   Home<self>
   Quick Start<quickstart>

.. toctree::
   :maxdepth: 1

   Electricity System Modeling<electricity_system_modeling/index>


.. toctree::
   :maxdepth: 1
   
   Network Utility Maximization<num/index>
