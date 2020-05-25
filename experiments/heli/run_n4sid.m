clear
close all
clc

%% N4SID test

dt = 0.02;

cd ../..

load('./datasets/split_normalized/n4siddata.mat');

nx = 10;

rng(1);

for trial = 0:10
    
    % prepare dataset
    inds = randperm(466);

    for i = 1:100
        ind = inds(i);
        y = squeeze(targets(ind,:,:));
        u = squeeze(inputs(ind,:,:));
        if i == 1
            dat = iddata(y,u,dt);
        else
            dat = merge(dat, iddata(y,u,dt));
        end
    end
    
    % run N4SID

    tic
    options = n4sidOptions('Display', 'on');
    sys = n4sid(dat, nx, options);
    tEnd = toc;
    fprintf('%d minutes and %f seconds\n', floor(tEnd/60), rem(tEnd,60));

    A = sys.A;
    B = sys.B;
    C = sys.C;
    D = sys.D;
    
    savename = sprintf('trained_models/SID/SID_10D_%d.mat',trial);

    save(savename,'A','B','C','D');
    
end