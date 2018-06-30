function opts = get_opts(opts, dataset, nbits, modelType, varargin)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generic
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ip = inputParser;
ip.addRequired('dataset'   , @isstr);
ip.addRequired('nbits'     , @isscalar);
ip.addRequired('modelType' , @isstr);

% model params
ip.addParameter('gamma', [1 0 0]);  % steepness of tanh relaxation 
% 2nd elem: =0 no annealing, >0 anneal freq. 3rd elem: scaling factor 
% e.g. [1 10 2]: initial gamma=1, x2 every 10 epochs
%
ip.addParameter('nbins', nbits);    % # bins in differentiable histogram binning
% the distance histogram has (nbins+1) bins, by default (nbits+1)
% use nbins<nbits for less sparse histograms
%
ip.addParameter('delta', 1);        % scaling factor for the \Delta parameter 
% usually we use delta=1. (optional) use delta>1 to "smooth" the gradients

% feature params
ip.addParameter('normalize', true);

% SGD params
ip.addParameter('batchSize' , 256);   % SGD batch size
ip.addParameter('lr'        , 0.1);   % base LR
ip.addParameter('lrdecay'   , 0.5);   % LR decay factor
ip.addParameter('lrstep'    , 10);    % decay LR every {lrstep} epochs
ip.addParameter('lrmult'    , 0.01);  % LR multiplier for pretrained layers
ip.addParameter('wdecay'    , 5e-4);  % weight decay
ip.addParameter('dropout'   , 0);     % dropout rate, 0 to disable

% train params
ip.addParameter('epochs'   , 50);     % total # epochs
ip.addParameter('gpus'     , []);     % which GPU(s) to use
ip.addParameter('continue' , true);   % continue from saved checkpoint?
ip.addParameter('debug'    , false);  % debug mode?

% test params
ip.addParameter('testInterval', 20);  % test every {testInterval} epochs

% misc params
ip.addParameter('randseed', 42);
ip.addParameter('prefix', []);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% parse input
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ip.KeepUnmatched = true;
ip.parse(dataset, nbits, modelType, varargin{:});
opts = catstruct(ip.Results, opts);  % combine w/ existing opts

end
