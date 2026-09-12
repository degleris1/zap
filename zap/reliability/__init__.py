"""Reliability modelling for capacity expansion planning (CH3).

Holds the thermal / storage forced-outage machinery: a versioned unit-key scheme
(``keys``), an on-demand two-state Markov sampler with an in-process slot cache
(``outages``), and the UCAP table derived from it. There is no outage store:
draws are a pure function of ``(base_seed, scheme, year, draw, key)``.

Import the submodule directly (``from zap.reliability import outages``); this
package deliberately does not re-export its contents so that
``python -m zap.reliability.outages`` does not execute the module twice.
"""
