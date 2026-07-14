function params = pack_polytope_params(A, b, beta, Lscale)
%PACK_POLYTOPE_PARAMS Pack polytope half-space data into the params layout
%the C++/MEX solver expects for shape_id = 2 (smooth polytope).
%
%   params = pack_polytope_params(A, b, beta, Lscale)
%
%   A      : m x 3 half-space normals (rows need not be pre-normalized;
%            this normalizes each row to unit length, same as
%            idcol::make_poly in core/idcol_implicitfamily.hpp)
%   b      : m x 1 half-space offsets, a_i'*y <= b_i
%   beta   : smooth-max sharpness
%   Lscale : length scale for the smooth-max (optional; defaults to
%            max(abs(b)) after normalization, matching the C++ default)
%
%   Layout: [beta; m; Lscale; A(:); b], size 3 + 4*m. A(:) is column-major
%   (MATLAB's native layout), matching pack_A_colmajor in
%   core/idcol_implicitfamily.hpp.

    b = b(:);
    m = size(A,1);
    if size(A,2) ~= 3 || numel(b) ~= m
        error('pack_polytope_params:badSize', 'A must be m x 3 and b must have m elements.');
    end

    row_norm = sqrt(sum(A.^2,2));
    if any(~isfinite(row_norm)) || any(row_norm <= 0)
        error('pack_polytope_params:badRow', 'A has a zero or invalid row normal.');
    end
    A = A ./ row_norm;
    b = b ./ row_norm;

    if nargin < 4 || isempty(Lscale) || ~(Lscale > 0)
        Lscale = max(abs(b));
        if Lscale == 0, Lscale = 1.0; end
    end

    params = zeros(3 + 4*m, 1);
    params(1) = beta;
    params(2) = m;
    params(3) = Lscale;
    params(4:3+3*m)     = A(:);   % column-major, matches Eigen's A(:) packing
    params(4+3*m:3+4*m) = b;
end
