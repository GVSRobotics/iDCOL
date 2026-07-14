%% demo_idcol_from_matlab.m
clear; clc;
% --- ensure iDCOL MEX is on path (local, non-global) ---
thisFile = mfilename('fullpath');
thisDir  = fileparts(thisFile);
idcolDir = fullfile(thisDir, '..');        % matlab_examples -> iDCOL root
mexDir   = fullfile(idcolDir, 'mex');
addpath(mexDir);
%% ----------------- Relative poses -----------------


g = [0.821168,   0.557509,  -0.121927,  -1.82107;
        0.123649,  0.0347633,   0.991717,  -2.76868;
        0.557129,  -0.829443,  -0.0403888, -0.302319;
        0, 0, 0, 1];


%% ----------------- Polytope (A1,b1) -----------------
A1 = [  1,  1,  1;
        1, -1, -1;
       -1,  1, -1;
       -1, -1,  1;
       -1, -1, -1;
       -1,  1,  1;
        1, -1,  1;
        1,  1, -1 ];
A1 = A1 * (1.0 / sqrt(3.0));

b1 = [1;1;1;1; 5/3;5/3;5/3;5/3];
Lscale = 1;
beta = 20.0;
n    = 8;

% NOTE: you need a MATLAB-side packer that matches your C++ packing.
% If you already have it as MEX or MATLAB function, call it here.
% Example signature:
%   params_poly = pack_polytope_params_rowmajor_mex(A1, b1, beta);
%
% If you DON'T have it yet, you must implement it (or hardcode for now).
params_poly = pack_polytope_params(A1, b1, beta, Lscale);
%% ----------------- Truncated Cone -----------------
rb = 1.0; rt = 1.5; ac = 1.5; bc = 1.5;
params_tc = [beta; rb; rt; ac; bc];
%% ----------------- Superellipsoid -----------------
a = 0.5; b = 1.0; c = 1.5;
params_se = [n; a; b; c];

%% ----------------- Superelliptic Cylinder -----------------
r = 1.0; h = 2.0; % half-height
params_sec = [n; r; h];

%% ----------------- Radial bounds options -----------------
optr = struct();
optr.num_starts = 1000;

% shape_id mapping from your C++ (core/shape_core.cpp):
% 1 sphere, 2 polytope, 3 truncated cone, 4 superellipsoid, 5 superelliptic cylinder
bounds_poly = radial_bounds_mex(2, params_poly, optr);
bounds_tc   = radial_bounds_mex(3, params_tc,   optr);
bounds_se   = radial_bounds_mex(4, params_se,   optr);
bounds_sec  = radial_bounds_mex(5, params_sec,  optr);

%% ----------------- ProblemData P -----------------
P = struct();
P.g = g;
P.shape_id1 = 2;
P.shape_id2 = 2;
P.params1 = params_poly;
P.params2 = params_poly;

%% ----------------- SolveData S (your new API) -----------------
S = struct();
S.P = P;
S.bounds1 = bounds_poly;
S.bounds2 = bounds_poly;

%% ----------------- Call iDCOL solver -----------------
out = idcol_solve_mex(S);

%% ----------------- Extract & post-check -----------------
x_star       = out.x;         % 3x1
alpha_star   = out.alpha;
lambda1 = out.lambda1;
lambda2 = out.lambda2;

% grad ordering is [dphi/dx; dphi/dalpha], so command uses global_xa_*
chk = shape_core_mex('global_xa_phi_grad', eye(4), x_star, alpha_star, P.shape_id1, P.params1);
phi_star  = chk.phi;
grad_star = chk.grad;     % 4x1
normal    = grad_star(1:3);

%% ----------------- Print -----------------
if out.converged
    fprintf('[iDCOL] converged = 1\n');
    fprintf('        fS_used = %.0f\n', out.fS_used);
    fprintf('        fS_attempts = %.0f\n', out.fS_attempts_used);
    fprintf('        iters = %.0f\n', out.iters_used);

    fprintf('        F =\n');
    disp(out.F);
    fprintf('        ||F|| = %.3e\n', out.final_F_norm);

    fprintf('        J =\n');
    disp(out.J);
    
    fprintf('        alpha = %.16g\n', alpha_star);
    fprintf('        x = [%.16g %.16g %.16g]\n', x_star(1), x_star(2), x_star(3));
    fprintf('        lambda1 = %.16g\n', lambda1);
    fprintf('        lambda2 = %.16g\n', lambda2);
    fprintf('        normal = [%.16g %.16g %.16g]\n', normal(1), normal(2), normal(3));
else
    fprintf('[iDCOL] converged = 0\n');
    fprintf('        fS_used = %.0f\n', out.fS_used);
    fprintf('        fS_attempts = %.0f\n', out.fS_attempts_used);
    fprintf('        iters = %.0f\n', out.iters_used);
    fprintf('        ||F|| = %.3e\n', out.final_F_norm);
    fprintf('        msg = %s\n', out.message);
end

%% Plotting

plotit = true;


if plotit
    
    phi1 = @(x,alpha) idcol_mex_phi_only_global(x, eye(4), alpha, P.shape_id1, P.params1);
    phi2 = @(x,alpha) idcol_mex_phi_only_global(x, g, alpha, P.shape_id2, P.params2);
    
    figure(1); clf;
    ax = axes('Parent', gcf); hold(ax,'on'); grid(ax,'on'); axis(ax,'equal'); view(ax,3);
    
    xyzlim = 5;
    
    h1 = plot_implicit_surface(phi1, 1.0, [-xyzlim xyzlim; -xyzlim xyzlim; -xyzlim xyzlim], 120, 0, ax);
    set(h1,'FaceColor',[0 0.6 1]);
    
    h2 = plot_implicit_surface(phi2, 1.0, [-xyzlim xyzlim; -xyzlim xyzlim; -xyzlim xyzlim], 120, 0, ax);
    set(h2,'FaceColor',[1 0.5 0]);
    
    %scaled
    h3 = plot_implicit_surface(phi1, alpha_star, [-xyzlim xyzlim; -xyzlim xyzlim; -xyzlim xyzlim], 120, 0, ax);
    set(h1,'FaceColor','r');
    
    h4 = plot_implicit_surface(phi2, alpha_star, [-xyzlim xyzlim; -xyzlim xyzlim; -xyzlim xyzlim], 120, 0, ax);
    set(h2,'FaceColor','g');
    
    %kissing point
    plot3(x_star(1), x_star(2), x_star(3), 'r.', 'MarkerSize', 25);
    
    % Axis labels (LaTeX)
    xlabel(ax,'$x\;(\mathrm{m})$','Interpreter','latex');
    ylabel(ax,'$y\;(\mathrm{m})$','Interpreter','latex');
    zlabel(ax,'$z\;(\mathrm{m})$','Interpreter','latex');
    
    
    
    % Title based on alpha_star (LaTeX)
    tol = 1e-6;
    if alpha_star > 1 + tol
        contactStr = '(Separated)';
    elseif abs(alpha_star - 1) <= tol
        contactStr = '(Contact)';
    else
        contactStr = '(Penetration)';
    end
    
    tstr = sprintf('$\\alpha^* = %.4f\\quad %s$', alpha_star, contactStr);
    title(ax, tstr, 'Interpreter','latex');
    
    
    axis tight
    lighting phong; camlight headlight; material dull
    set(gcf,'Renderer','opengl')
    
end

function params = pack_polytope_params(A, b, beta, A_scale)
% params = [beta; m; A_scale; A(:); b]
% where A(:) is MATLAB column-major (same as your C++ loop j*m+i).

    if nargin < 4
        A_scale = 1; % you used 1 in C++
    end

    [m, n] = size(A);
    if n ~= 3
        error('A must be m x 3');
    end
    if numel(b) ~= m
        error('b must be length m');
    end

    params = zeros(3 + 3*m + m, 1);
    params(1) = beta;
    params(2) = m;
    params(3) = A_scale;

    params(4 : 3 + 3*m) = A(:);         % column-major, matches your C++ packing
    params(3 + 3*m + 1 : end) = b(:);   % ensure column
end
