#!/bin/bash
while getopts f:c: flag
do
    case "${flag}" in
        f) fsldir=${OPTARG};;
        c) cmd=${OPTARG};;
    esac
done

FSLDIR=$fsldir
. ${FSLDIR}/etc/fslconf/fsl.sh
PATH=${FSLDIR}/bin:${PATH}
export FSLDIR PATH
$cmd