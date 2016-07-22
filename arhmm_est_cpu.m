%{

Parallel MCMC sampling of AR-HMMs for stochastic time series prediction.
 - Calculate predictions for a given set of AR-HMMs and an initial observation sequence.

Written by I. R. Sipos (siposr@hit.bme.hu) and A. Ceffer (ceffer@hit.bme.hu)
(Department of Networked Systems and Services,
Budapest University of Technology and Economics, Budapest, Hungary, 2016.)

Intended for academic purposes on an as is basis. Any feedback is appreciated.
Please refer to:

(1) SIPOS, I. Róbert. Parallel stratified MCMC sampling of AR-HMMs for stochastic time series prediction.
In: Proceedings, 4th Stochastic Modeling Techniques and Data Analysis (SMTDA2016). Valletta, 2016.

(2) SIPOS, I. Róbert; CEFFER, Attila; LEVENDOVSZKY, János. Parallel optimization of sparse portfolios with AR-HMMs.
Computational Economics, Online First. DOI: 10.1007/s10614-016-9579-y

%}

function [pred, LL] = arhmm_est_cpu(hmms, x)

    warning('off');
    addpath(genpath('HMMall'));
    warning('on');

    % Initializations.
    K = size(hmms, 2);
    pred = nan(1, K);
    LL = nan(1, K);
    T = length(x);
    
    % Sequentially iterate through each AR-HMM.
    for k = 1:K
        hmm = hmms(:, k);
        N = length(hmm.pi);
        
        % Observation likelihood for time instance and each state.
        obslik = ones(N, T-1)*eps;
        for i = 1:N
            for t = 1:T-1
                if hmm.sigma(i) > 0
                    obslik(i, t) = normpdf(x(t+1)-hmm.rho(i)*x(t), hmm.mu(i), hmm.sigma(i));
                end
            end
        end
        
        % Call the forward-backward algorithm.
        [~, ~, gamma, loglik] = fwdback(hmm.pi, hmm.A, obslik);
        
        % Prediction.
        alpha = gamma(:, end).' * hmm.A;
        y = alpha * (hmm.rho.*x(end) + hmm.mu);
        
        % Store the results.
        pred(k) = y;
        LL(k) =  loglik;
    end
end
