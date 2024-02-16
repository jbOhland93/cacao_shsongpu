# Types of scripts

This folder contains scripts to set up and control the ARTAO loop. In general, there are two types: aorun and qrun scripts.

## aorun scripts
The scripts with the aorun prefix are the primary way to set up the loop step by step and make it easy to change parameters and find issues in the process. These scripts are recommended, especially when learning how to use ARTAO.

## qrun scripts
The qrun ("quick-run") scripts bundle steps that are usually done in succession while setting up and running the ARTAO loop. This is for easier and faster operation, NOT for learning how the loop works.
If problems are encountered here, it is useful to step back and use the fine-grained aorun scripts in order to find the issue.

Furthermore, it is noteworthy that the qrun scripts rely on third party software, liky pyMilk and ds9. These have to be installed.
