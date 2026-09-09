"""Reliability modelling for capacity expansion planning (CH3).

Currently holds the thermal / storage forced-outage machinery: a virtual unit
pool derived from the static component tables, a two-state Markov sampler, an
on-disk outage store, and the UCAP table derived from it.

Import the submodule directly (``from zap.reliability import outages``); this
package deliberately does not re-export its contents so that
``python -m zap.reliability.outages`` does not execute the module twice.
"""
