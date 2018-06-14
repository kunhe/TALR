addpath('main');
addpath('util');

run ./matconvnet/matlab/vl_setupnn
myLogInfo('[MatConvNet] ready');

run vlfeat/toolbox/vl_setup
myLogInfo('[VLFeat] ready');

myLogInfo('TALR initialized!');
