% Image dimension
n = 256; 
% Blur radius
r = 7; 

% Construct blurr H >> Toeplitz 
H1 = toeplitz([1/(2*r-1)*ones(1,r), zeros(1,n-r)]);
H2 = H1; 

% Use reshape to avoid large Kronecker products
H = @(x) reshape(H2 * reshape(H1 * reshape(x, n, n)', n, n)', n^2, 1);
HT = @(x) reshape(H1' * reshape(H2' * reshape(x, n, n)', n, n)', n^2, 1);

% Image (cameraman from Matlab)
X_original = im2double(imread('cameraman.tif'));

% Blurred and noisy image 
x_true = X_original(:);
g_true = H(x_true);
noise = randn(size(g_true)) * 0.0001 * norm(g_true) / norm(randn(size(g_true)));
g = g_true + noise;

% Reshape for visualization
G_noisy = reshape(g, n, n);

% L-Curve to find optimal lambda 
lambda_values = logspace(-6, -2, 100);
residuals = zeros(size(lambda_values));
regularizations = zeros(size(lambda_values));

% Global-gmres
max_iter = 5; 
restart = 20; 
X0 = zeros(n^2, 1);
for i = 1:length(lambda_values)
    lambda = lambda_values(i);
    A_reg = @(x) HT(H(x)) + lambda^2 * x;
    [x_reg, ~] = gmres(A_reg, HT(g), restart, 1e-6, max_iter);
    residuals(i) = norm(H(x_reg) - g);
    regularizations(i) = norm(x_reg);
end

% Plot L-curve
figure;
loglog(residuals, regularizations, '-o');
xlabel('Residual Norm ||Hx-g||_2');
ylabel('Regularization Norm ||x||_2');
title('L-Curve');

% Find optimal lambda using maximum curvature
[~, idx_opt] = min(abs(diff(diff(log(residuals)))));
lambda_opt = lambda_values(idx_opt);

% Final restoration
A_opt = @(x) HT(H(x)) + lambda_opt^2 * x;
[X_restored, ~] = gmres(A_opt, HT(g), restart, 1e-6, max_iter);
X_restored = reshape(X_restored, n, n);

% Display results
figure;
subplot(1,3,1), imshow(X_original), title('Original Image');
subplot(1,3,2), imshow(G_noisy), title('Blurred & Noisy Image');
subplot(1,3,3), imshow(X_restored), title('Restored Image');
