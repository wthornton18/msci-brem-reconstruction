#!/bin/bash

for VARIABLE in 1 2 3 4
do   
  export CURRENT_VAR=$VARIABLE
  printenv
done 
