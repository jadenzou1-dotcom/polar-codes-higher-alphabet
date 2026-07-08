# Polar Codes for Higher-Order Alphabets

Simulating polar codes over q-ary symmetric channels to test whether channel
polarization persists at higher alphabet sizes.

**Author:** Jaden Zou · Cornell MEng ECE · Advisor: Prof. Aaron Wagner
[LinkedIn](https://www.linkedin.com/in/jadenzou55/)

## Summary

This project implements a full polar code simulator — encoder, successive
cancellation decoder, and sub-channel construction — starting from the binary
erasure channel (BEC), extending to the binary symmetric channel (BSC), and
finally generalizing to q-ary symmetric channels for q ∈ {2, 3, 5, 7}.

The central question: does channel polarization — the phenomenon that makes
polar codes capacity-achieving in the binary case — persist as the alphabet
size grows? The answer is yes, and more: polarization actually **sharpens**
as q increases, an unexpected result.

## Background

Polar codes (Arıkan, 2009) are the first provably capacity-achieving codes
with efficient O(N log N) encoding and decoding. They work by transforming N
uses of a channel into N synthetic sub-channels that "polarize" — some become
nearly perfect, the rest nearly useless. This project asks whether that
behavior holds beyond the binary case.

## Approach

`baseline/` holds the binary implementation: BEC first, where sub-channel
reliability can be computed analytically via the Bhattacharyya recursion,
then BSC, where no closed-form recursion exists so reliability is estimated
via Monte Carlo simulation instead.

`higher_alphabet/` generalizes the polar transform, channel model, and
decoder to q-ary symbols (q ∈ {2, 3, 5, 7}). Comparing raw block error rate
across alphabet sizes turned out to be misleading (larger q is unfairly
favored), which motivated switching to **normalized subchannel capacity** as
the real comparison metric — the result in `plot_normalized_capacity.py` is
the main finding of the project.

## Full report and poster

`jz2396_MEng_Project_Report.pdf` is the complete written report submitted for
Cornell's MEng program, and `jadenpolarposter.pdf` is the accompanying
presentation poster. The report is organized by semester rather than by this
repo's layout, so this README is the better starting point if you just want
the gist of the project.

## Tech stack

Python, NumPy, Matplotlib

## Note on the code

Some files still contain leftover print statements and debug output from
development — these were left in in case they're useful for
tracing through the pipeline again, and don't affect correctness. Everything
runs fine as-is.
