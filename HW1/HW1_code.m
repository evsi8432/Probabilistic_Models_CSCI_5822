
%%% Task 1 %%%

% create values
i = 1:10;
prior6 = prior(i,[6;6]);
prior12 = prior(i,[12;12]);

% make plots
figure(1)

subplot(2,1,1)
bar(1:10,prior6)
title("Prior Distribution for \sigma_1 = \sigma_2 = 6")
xlabel("h")
ylabel("Probability Mass")
ylim([0,0.5])

subplot(2,1,2)
bar(1:10,prior12)
title("Prior Distribution for \sigma_1 = \sigma_2 = 12")
xlabel("h")
ylabel("Probability Mass")
ylim([0,0.5])


%%% Task 2 %%%

x = 0.5;
y = 1.5;

post12 = posterior(i,[12;12],x,y);

figure(2)
bar(1:10,post12)
title("Posterior Probabilities for \sigma_1 = \sigma_2 = 12, X = {(1.5,0.5)}")
xlabel("h")
ylabel("Probability Mass")

%%% Task 3 %%%

rng = -10:0.1:10; 
[x0,y0] = meshgrid(rng,rng);
figure(3)
plot_Q(i,[10;10],x,y,x0,y0)
title("P(Y \in concept) Over 2-D Space, X = (1.5,0.5)")

%%% Task 4 %%%

x = 4.5;
y = 2.5;
figure(4)
plot_Q(i,[10;10],x,y,x0,y0)
title("P(Y \in concept) Over 2-D Space, X = (4.5,2.5)")

%%% Task 5 %%%

x = [2.2,0.5,1.5];
y = [-0.2,0.5,1];

figure(5)

subplot(1,3,1)
plot_Q(i,[30;30],x(1),y(1),x0,y0)

subplot(1,3,2)
plot_Q(i,[30;30],x(1:2),y(1:2),x0,y0)
title("P(Y \in concept) Over 2-D Space")

subplot(1,3,3)
plot_Q(i,[30;30],x,y,x0,y0)


%%% Task 6 %%%

% experiement with a uniform prior

figure(6)

subplot(1,3,1)
plot_Q_uniform(i,x(1),y(1),x0,y0)

subplot(1,3,2)
plot_Q_uniform(i,x(1:2),y(1:2),x0,y0)
title("P(Y \in concept) Over 2-D Space with Uniform Prior")

subplot(1,3,3)
plot_Q_uniform(i,x,y,x0,y0)

% make hypothesis class a beta distributions

x = [2.2,0.5,1.5];
y = [-0.2,0.5,0.99];

figure(7)

b = 4;

subplot(1,3,1)
plot_Q_B(i,[30;30],x(1),y(1),x0,y0,b)

subplot(1,3,2)
plot_Q_B(i,[30;30],x(1:2),y(1:2),x0,y0,b)
title("P(Y \in concept) Over 2-D Space with Beta Likelihood, a = b = 4")

subplot(1,3,3)
plot_Q_B(i,[30;30],x,y,x0,y0,b)

figure(8)

subplot(3,1,1)
postB = posterior_B(i,[30;30],x,y,0.5);
bar(1:10,postB)
title("Posterior with Beta Likelihood, a = b = 0.5")
xlabel("h")
ylabel("Probability Mass")

subplot(3,1,2)
postB = posterior_B(i,[30;30],x,y,2);
bar(1:10,postB)
title("Posterior with Beta Likelihood, a = b = 2")
xlabel("h")
ylabel("Probability Mass")

subplot(3,1,3)
postB = posterior_B(i,[30;30],x,y,4);
bar(1:10,postB)
title("Posterior with Beta Likelihood, a = b = 4")
xlabel("h")
ylabel("Probability Mass")







%%% Helper Functions %%%

% define the prior
function p = prior(i,sig)
    s = [2.*i;2.*i];
    p = exp(-sum(s./sig));
    p = p./sum(p);
end

% define the likelihood
function L = likelihood(x,y,i)
    num_hyp = numel(i);    
    L = zeros(1,num_hyp);
    for j = 1:num_hyp
        L(j) = prod(((abs(x) <= i(j)) & (abs(y) <= i(j)))./(4*i(j)^2));
    end
end

% define the posterior
function P = posterior(i,sig,x,y)
    p = prior(i,sig);
    L = likelihood(x,y,i);
    P = p.*L;
    P = P./sum(P);
end

% define the probability that point y
% is in the hypothesis space
function P = Q_prob(i,sig,x,y,x0,y0)
    P = zeros(numel(x0),1); 
    hypo_probs = posterior(i,sig,x,y);
    for j = 1:numel(x0)
        P(j) = sum(hypo_probs .* ...
        ((abs(x0(j)) <= i) & (abs(y0(j)) <= i)));
    end
end

% plot the results of the function above
function plot_Q(i,sig,x,y,x0,y0)
    Q_probs = Q_prob(i,sig,x,y,x0,y0);
    Q_probs = vec2mat(Q_probs',201);
    contourf(x0,y0,Q_probs,0:0.01:1)
    h = contourcbar;
    h.YLabel.String = "Probability";
    h.YLim = [0,1];
    set(gca, 'CLim', [0 1])
    hold on
    plot(x,y,'k*','Markersize',10)
    xticks(-10:2:10);
    yticks(-10:2:10);
end

% define the prior for task 6
function p = uniform_prior(i)
    p = 1/numel(i);
end


% define the likelihood for task 6
function L = likelihood_B(x,y,i,b)
    num_hyp = numel(i);    
    L = zeros(1,num_hyp);
    for j = 1:num_hyp
        xt = 0.5 + x./(2*i(j));
        yt = 0.5 + y./(2*i(j));
        L(j) = prod(betapdf(xt,b,b).*...
            betapdf(yt,b,b)./(4*i(j)^2));
    end
end

% define the posteriors for task 6
function P = uniform_posterior(i,x,y)
    p = uniform_prior(i);
    L = likelihood(x,y,i);
    P = p.*L;
    P = P./sum(P);
end

function P = posterior_B(i,sig,x,y,b)
    p = prior(i,sig);
    L = likelihood_B(x,y,i,b);
    P = p.*L;
    P = P./sum(P);
end

% Q_probs for uniform
function P = Q_prob_uniform(i,x,y,x0,y0)
    P = zeros(numel(x0),1); 
    hypo_probs = uniform_posterior(i,x,y);
    for j = 1:numel(x0)
        P(j) = sum(hypo_probs .* ...
        ((abs(x0(j)) <= i) & (abs(y0(j)) <= i)));
    end
end

% Q_probs for beta
function P = Q_prob_B(i,sig,x,y,x0,y0,b)
    P = zeros(numel(x0),1); 
    hypo_probs = posterior_B(i,sig,x,y,b);
    for j = 1:numel(x0)
        P(j) = sum(hypo_probs .* ...
        ((abs(x0(j)) <= i) & (abs(y0(j)) <= i)));
    end
end

% plot the results of the function above
function plot_Q_uniform(i,x,y,x0,y0)
    Q_probs = Q_prob_uniform(i,x,y,x0,y0);
    Q_probs = vec2mat(Q_probs',201);
    contourf(x0,y0,Q_probs,0:0.01:1)
    h = contourcbar;
    h.YLabel.String = "Probability";
    h.YLim = [0,1];
    set(gca, 'CLim', [0 1]);
    hold on
    plot(x,y,'k*','Markersize',10)
    xticks(-10:2:10);
    yticks(-10:2:10);
end

% plot the results of the function above
function plot_Q_B(i,sig,x,y,x0,y0,b)
    Q_probs = Q_prob_B(i,sig,x,y,x0,y0,b);
    Q_probs = vec2mat(Q_probs',201);
    contourf(x0,y0,Q_probs,0:0.01:1)
    h = contourcbar;
    h.YLabel.String = "Probability";
    h.YLim = [0,1];
    set(gca, 'CLim', [0 1]);
    hold on
    plot(x,y,'k*','Markersize',10)
    xticks(-10:2:10);
    yticks(-10:2:10);
end