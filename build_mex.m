function build_mex()
    thisFile = mfilename('fullpath');
    rootDir  = fileparts(thisFile);

    eigenDir = getenv('EIGEN3_INCLUDE_DIR');
    if isempty(eigenDir)
        eigenDir = 'C:/vcpkg/installed/x64-windows/include/eigen3';
    end

    inc1  = ['-I' rootDir];
    inc2  = ['-I' eigenDir];

    outd = fullfile(rootDir,'mex');
    if ~exist(outd,'dir'), mkdir(outd); end

    % C++17 flag (compiler-dependent)
    if ispc
        cxx17 = 'COMPFLAGS=$COMPFLAGS /std:c++17';
    else
        cxx17 = 'CXXFLAGS=$CXXFLAGS -std=c++17';
    end

    mex('-O','-outdir',outd,inc1,inc2,cxx17, ...
        fullfile('mex','idcol_solve_mex.cpp'), ...
        fullfile('core','shape_core.cpp'), ...
        fullfile('core','idcol_kkt.cpp'), ...
        fullfile('core','idcol_newton.cpp'), ...
        fullfile('core','idcol_solve.cpp'));


    mex('-O','-outdir',outd,inc1,inc2,cxx17, ...
        fullfile('mex','idcol_kkt_mex.cpp'), ...
        fullfile('core','shape_core.cpp'), ...
        fullfile('core','idcol_kkt.cpp'));

    mex('-O','-outdir',outd,inc1,inc2,cxx17, ...
        fullfile('mex','shape_core_mex.cpp'), ...
        fullfile('core','shape_core.cpp'));

    mex('-O','-outdir',outd,inc1,inc2,cxx17, ...
        fullfile('mex','radial_bounds_mex.cpp'), ...
        fullfile('core','shape_core.cpp'), ...
        fullfile('core','radial_bounds.cpp'));

    fprintf('[iDCOL] MEX build complete\n');
end
