#!/bin/bash
echo "Launching Julia files"
nameFile="_SIMUL_1"
nCores=10
nThreads=$nCores
dateString=$(date +'%d-%m-%Y_%H.%M')

mkdir -p Outputs
nohup julia -p $nCores -t $nThreads "Nonlinear Dephasing - Launcher.jl" $nameFile > "Outputs/out$nameFile""_p$nCores""_t$nThreads""_$dateString.out" &

echo "Launching completed"




