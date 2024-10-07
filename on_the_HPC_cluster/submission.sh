#!/bin/bash                                                                                        
#PBS -l select=1:ncpus=2:mem=3gb                                                                   
#PBS -l walltime=24:00:00                                                                          

module load julia/1.6.4
cd $HOME/Graphon
#julia Integrate_Periodic_Potential.jl ./parameters.csv  $index  