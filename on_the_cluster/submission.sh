#!/bin/bash
#PBS -N Kuramoto_on_ER
#PBS -m ea
#PBS -q medium

cd ${HOME}/project_Greg
/usr/local/bin/julia Kuramoto.jl ./parameters.csv  $index


