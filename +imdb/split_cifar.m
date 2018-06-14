function S = cifar_split(Y, opts)
if opts.split == 1
    trainPerCls = 500; testPerCls = 100;
else, assert(opts.split == 2);
    trainPerCls = 0; testPerCls = 1000;
end

if ~iscolumn(Y), Y = Y'; end
assert(size(Y, 1) == 60e3);
S = 2 * ones(size(Y, 1), 1);  % default = val(2)

for c = 1:10
    % use the first testPerCls for test, next trainPerCls for train
    % but if trainPerCls<=0, use the rest for train
    ind = find(Y == c);
    ind = ind(randperm(length(ind)));
    assert(length(ind) >= trainPerCls + testPerCls);

    itest = ind(1:testPerCls);
    if trainPerCls > 0
        itrain = ind(testPerCls+1:trainPerCls+testPerCls);
    else
        itrain = ind(testPerCls+1:end);
    end
    S(itest) = 3;
    S(itrain) = 1;
end
end
