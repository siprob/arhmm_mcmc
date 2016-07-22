%{

Parallel MCMC sampling of AR-HMMs for stochastic time series prediction.

Written by I. R. Sipos (siposr@hit.bme.hu) and A. Ceffer (ceffer@hit.bme.hu)
(Department of Networked Systems and Services,
Budapest University of Technology and Economics, Budapest, Hungary, 2016.)

Intended for academic purposes on an as is basis. Any feedback is appreciated.
Please refer to:

(1) SIPOS, I. Róbert. Parallel stratified MCMC sampling of AR-HMMs for stochastic time series prediction.
In: Proceedings, 4th Stochastic Modeling Techniques and Data Analysis (SMTDA2016). Valletta, 2016.

(2) SIPOS, I. Róbert; CEFFER, Attila; LEVENDOVSZKY, János. Parallel optimization of sparse portfolios with AR-HMMs.
Computational Economics, Online First. DOI: 10.1007/s10614-016-9579-y

An nVidia compute capability 2.0 or higher device is required to enable the GPU.

%}

clear;

% Parameters.
options = struct(...
    'nofStates', 8, ... % Defines the maximum number of hidden states, AR-HMMs with 1..nofStates states will be considered.
    'parallelMCMC', 4, ... % Gives the number of parallel MCMC threads per a certain number of hidden states.
    'samples', 500, ... % The number of samples to take.
    'burnin', 100, ... % The first burnin number of samples will be discarded from the prediction.
    'jumpfactor', 1/3, ... % Controls how "far" the next sample can be from the previous. Greater value implies greater jumps in the model space.
    'GPUEnabled', false, ... % Whether or not the computation should take place on a GPU.
    'maxHMMperBlocks', 128 ... % In case of using a GPU, it gives the number of AR-HMMs to assign to each multiprocessor block.
);
GPU = struct();

% Load the observed time series (for GPU, its length must be a multiple of 32 + 1).
x = 100 + randn(1, 33);

% -----------------------------------------------------------------------------

% Seed the random generator.
seed = RandStream('mcg16807','Seed', 1986);
RandStream.setGlobalStream(seed);

% Initial AR-HMM construction.
hmms = arhmm_init(x, options);

% MCMC sampling.
LLtrace = nan(options.samples, length(hmms));
predtrace = nan(options.samples, length(hmms));
LL = -Inf * ones(1, length(hmms));
for c = 1:options.samples

    % Generate new candidates.
    hmms1 = struct();
    for i = 1:length(hmms)
        hmms1new = arhmm_neighbor(hmms(i), options.jumpfactor);
        hmms1(i).N = hmms1new.N;
        hmms1(i).pi = hmms1new.pi;
        hmms1(i).A = hmms1new.A;
        hmms1(i).mu = hmms1new.mu;
        hmms1(i).sigma = hmms1new.sigma;
        hmms1(i).rho = hmms1new.rho;
    end

    % Compute the predictions and their likelihoods for each AR-HMM candidate.
    if options.GPUEnabled
        [pred, LL1, GPU] = arhmm_est_gpu(hmms1, x.', options, GPU);
    else
        [pred, LL1] = arhmm_est_cpu(hmms1, x.');
    end

    % Metropolis-Hastings acceptance.
    for i = 1:length(hmms)
        if LL1(i) > LL(i) || rand < exp(LL1(i)-LL(i))
            LL(i) = LL1(i); LLtrace(c, i) = LL(i);
            hmms(i) = hmms1(i);
            predtrace(c, i) = pred(i);
        end
    end
end

% -----------------------------------------------------------------------------

% Process the results.

% Maximum likelihood estimate.
[~, idx] = max(LLtrace(:));
predML = predtrace(idx);

% MCMC estimate.
classpred = nan(options.nofStates, 1);
classLL = nan(options.nofStates, 1);
for i = 1:options.nofStates
    predset = predtrace(options.burnin+1:end, (i-1)*options.parallelMCMC+1:i*options.parallelMCMC);
    classpred(i) = nanmean(predset(:));

    LLset = LLtrace(options.burnin+1:end, (i-1)*options.parallelMCMC+1:i*options.parallelMCMC);
    LLset = LLset(~isnan(LLset));
    LLoffset = max(LLset);
    classLL(i) = LLoffset + log(sum(exp(LLset - LLoffset)));
end
classL = exp(classLL - max(classLL));
classP = classL ./ sum(classL);
predMCMC = classP.' * classpred;

fprintf('ML prediction: %f, MCMC prediction: %f\n', predML, predMCMC);
