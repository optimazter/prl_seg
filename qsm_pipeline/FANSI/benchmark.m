function times = benchmark(count, isGPU)
%BENCHMARK  FANSI Benchmark
%   BENCHMARK times six different FANSI tasks and compares the execution
%   speed with the speed of several other computers.  The five tasks are:
%
%    TKD      Thresholded K-space Division       Direct solver, FFTs,
%                                                thresholding
%    NDI      Nonlinear Dipole Inversion         Iterative solver, <10 FFTs
%                                                Gradient Descent iterations
%    nlTGV    Nonlinear Total Generalised        Iterative sovler, >20 FFTs
%             Variation                          ADMM iterations
%compute_xsim Susceptibility Similarity Metric   Compute 'error' metric
%  imagesc3d2 2D Plotting routine                Plotting used in
%                                                visualising iterations
%
%   A final bar chart shows speed, which is inversely proportional to
%   time.  Here, longer bars are faster machines, shorter bars are slower.
%
%   BENCHMARK runs each of the five tasks once.
%   BENCHMARK(N) runs each of the five tasks N times.
%   BENCHMARK(0) just displays the results from other machines.
%   T = BENCHMARK(N) returns an N-by-5 array with the execution times.
%
% Based on MATLABs builtin BENCH command
% Created by Patrick Fuchs (01.2022)

if nargin < 2, isGPU = 1; end
isGPU = isGPU && gpuDeviceCount;
if nargin < 1, count = 1; end
times = zeros(count,5);
fig1 = figure('WindowStyle','normal');
set(fig1,'pos','default','menubar','none','numbertitle','off', ...
    'name','FANSI Benchmark');
hax1 = axes('position',[0 0 1 1],'parent',fig1);
axis(hax1,'off');
text(.5,.6,'FANSI Benchmark','parent',hax1,'horizontalalignment','center','fontsize',18)
task = text(.50,.42,'','parent',hax1,'horizontalalignment','center','fontsize',18);
drawnow
pause(1);

% make sure required tools / datasets are in path
[p,~] = fileparts(mfilename('fullpath'));
addpath([p,filesep,'data_challenge']);
addpath([p,filesep,'Inversion',filesep,'ClosedForm']);
addpath([p,filesep,'Inversion',filesep,'eNDI']);

% Use a private stream to avoid resetting the global stream
stream = RandStream('mt19937ar');

problemsize = zeros(1, 4);

for k = 1:count
    % TKD n = 512x512x512 = 134217728
    set(task,'string','TKD')
    drawnow
    [times(k,1), problemsize(1)] = bench_TKD(stream);

    % NDI n = 256x256x256 = 16777216
    set(task,'string','NDI')
    drawnow
    [times(k,2), problemsize(2)] = bench_NDI(isGPU);

    % nlTGV n = 256x256x256 = 16777216
    set(task,'string','NDI')
    drawnow
    [times(k,3), problemsize(3)] = bench_nlTGV(isGPU);

    % compute_xsim n = 512x512x512 = 134217728
    set(task,'string','NDI')
    drawnow
    [times(k,4), problemsize(4)] = bench_compute_xsim(stream);

    % 2-D graphics
    set(task,'string','imagesc3d2')
    drawnow
    times(k,5) = bench_imagesc3d2('off');

end

if exist('benchmark_fansi.dat','file') ~= 2
    warning(message('MATLAB:bench:noDataFileFound'))
    return
end
fp = fopen('benchmark_fansi.dat', 'rt');

% Skip over headings in first three lines.
for k = 1:3
    fgetl(fp);
end

% Read the comparison data

specs = {};
T = [];
details = {};
g = fgetl(fp);
m = 0;
desclength = 75;
while length(g) > 1
    m = m+1;
    specs{m} = g(1:desclength); %#ok<AGROW>
    T(m,:) = sscanf(g((desclength+1):end),'%f')'; %#ok<AGROW>
    details{m} = fgetl(fp); %#ok<AGROW>
    g = fgetl(fp);
end

% Close the data file
fclose(fp);

% Determine the best 10 runs (if user asked for at least 10 runs)
if count > 10
    warning('Only using best 10/%i trials.', count);
    totaltimes = 100./sum(times, 2);
    [~, timeOrder] = sort(totaltimes, 'descend'); 
    selected = timeOrder(1:10);
else
    selected = 1:count;
end

meanValues = mean(T, 1);

% Add the current machine and sort
T = [T; times(selected, :)];
this = [zeros(m,1); ones(length(selected),1)];
if count==1
    % if a single BENCH run
    specs(m+1) = {['This machine', repmat(' ', 1, desclength-12)]};
    details{m+1} = ['Your machine running ', version];
else
    for k = m+1:size(T, 1)
        ind = k-m; % this varies 1:length(selected)
        sel = num2str(selected(ind));   
        specs(k) = {['This machine run ', sel, repmat(' ', 1, desclength-18-length(sel))]}; %#ok<AGROW>
        details{k} = ['Your machine running ', version, ', run', sel]; %#ok<AGROW> 
    end
end
scores = mean(bsxfun(@rdivide, T, meanValues), 2);
m = size(T, 1);

% Normalize by the sum of meanValues to bring the results in line with
% earlier implementation 
speeds = (100/sum(meanValues))./(scores);
[~,k] = sort(speeds);
specs = specs(k);
details = details(k);
T = T(k,:);
this = this(k);

clf(fig1)
set(fig1,'pos',get(fig1,'pos')+[50 -150 230 0]);

% Defining layout constants - change to adjust 'look and feel'
% The names of the tests
TestNames = {'TKD', 'NDI', 'nlTGV', 'XSim', 'imagesc3d2'};

testDatatips = {sprintf('TKD reconstruction of object with %ix%ix%i voxels',problemsize(1),problemsize(1),problemsize(1)),...
    sprintf('NDI reconstruction of object with %i voxels',problemsize(2)),...
    sprintf('nlTGV reconstruction of object with %i voxels',problemsize(3)),...
    sprintf('XSim computation of object with %ix%ix%i voxels',problemsize(4),problemsize(4),problemsize(4)),...
    'Loop over all slices of a susceptibility map using imagesc3d2'};
% Number of test columns
NumTests = size(TestNames, 2);
NumRows = m+1;      % Total number of rows - header (1) + number of results (m)
TopMargin = 0.05; % Margin between top of figure and title row
BotMargin = 0.20; % Margin between last test row and bottom of figure
LftMargin = 0.03; % Margin between left side of figure and Computer Name
RgtMargin = 0.03; % Margin between last test column and right side of figure
CNWidth = 0.40;  % Width of Computer Name column
MidMargin = 0.03; % Margin between Computer Name column and first test column
HBetween = 0.005; % Distance between two rows of tests
WBetween = 0.015; % Distance between two columns of tests
% Width of each test column
TestWidth = (1-LftMargin-CNWidth-MidMargin-RgtMargin-(NumTests-1)*WBetween)/NumTests;
% Height of each test row
RowHeight = (1-TopMargin-(NumRows-1)*HBetween-BotMargin)/NumRows;
% Beginning of first test column
BeginTestCol = LftMargin+CNWidth+MidMargin;
% Retrieve the background color for the figure
bc = get(fig1,'Color');
YourMachineColor = [0 0 1];

% Create headers

% Computer Name column header
uicontrol(fig1,'Style', 'text', 'Units', 'normalized', ...
    'Position', [LftMargin 1-TopMargin-RowHeight CNWidth RowHeight],...
    'String',  getString(message('MATLAB:bench:LabelComputerType')),...
    'BackgroundColor', bc, 'Tag', 'Computer_Name','FontWeight','bold');

% Test name column header
for k=1:NumTests
    uicontrol(fig1,'Style', 'text', 'Units', 'normalized', ...
        'Position', [BeginTestCol+(k-1)*(WBetween+TestWidth) 1-TopMargin-RowHeight TestWidth RowHeight],...
        'String', TestNames{k}, 'BackgroundColor', bc, 'Tag', TestNames{k}, 'FontWeight', 'bold', ...
        'TooltipString', testDatatips{k});
end
% For each computer
for k=1:NumRows-1
    VertPos = 1-TopMargin-k*(RowHeight+HBetween)-RowHeight;
    if this(NumRows - k)
        thecolor = YourMachineColor;
    else
        thecolor = [0 0 0];
    end
    % Computer Name row header
    uicontrol(fig1,'Style', 'text', 'Units', 'normalized', ...
        'Position', [LftMargin VertPos CNWidth RowHeight],...
        'String', specs{NumRows-k}, 'BackgroundColor', bc, 'Tag', specs{NumRows-k},...
        'TooltipString', details{NumRows-k}, 'HorizontalAlignment', 'left', ...
        'ForegroundColor', thecolor);
    % Test results for that computer
    for n=1:NumTests
        uicontrol(fig1,'Style', 'text', 'Units', 'normalized', ...
            'Position', [BeginTestCol+(n-1)*(WBetween+TestWidth) VertPos TestWidth RowHeight],...
            'String', sprintf('%.4f',T(NumRows-k, n)), 'BackgroundColor', bc, ...
            'Tag', sprintf('Test_%d_%d',NumRows-k,n), 'ForegroundColor', thecolor);
    end
end

% Warning text
uicontrol(fig1, 'Style', 'text', 'Units', 'normalized', ...
    'Position', [0.01 0.01 0.98 BotMargin-0.02], 'BackgroundColor', bc, 'Tag', 'Disclaimer', ...
    'String', 'Place the cursor near a computer name for system and version details.' );

set(fig1, 'NextPlot', 'new');

% Log selected bench data
logBenchData(times(selected, :),isGPU);

end

% ----------------------------------------------- %
function [t, n] = bench_TKD(stream)
n = 512;
N = [n,n,n];
dr = [1,1,1];

kernel = dipole_kernel_fansi( N, dr, 0 ); %#ok<NASGU> 


reset(stream,0);
phase = randn(stream, N);
mask = ones(size(phase)); %#ok<NASGU> 

kthre = 0.08; %#ok<NASGU> truncation threshold

tic; % evalc leads to a tremendous slowdown on this command
[~] =evalc('tkd(phase, mask, kernel, kthre, N);');
t = toc;
end
% ----------------------------------------------- %
function [t, n] = bench_NDI(gpuFlag)
magnitude = getfield(load('magn.mat'),'magn');
phase = getfield(load('phs_tissue.mat'),'phs_tissue');
mask = getfield(load('msk.mat'),'msk');
dr = getfield(load('spatial_res.mat'),'spatial_res');
n = size(mask);

kernel = dipole_kernel_fansi( n, dr, 0 );

TE = 25e-3;
B0 = 2.8936;
gyro = 2*pi*42.58;
phs_scale = TE * gyro * B0;
 
params = [];
params.input = phase * phs_scale;
params.weight = (magnitude .* mask);
params.K = kernel;
params.maxOuterIter = 250;
params.isGPU = gpuFlag;

tic % use evalc to capture command window output of function and void
[~] = evalc('ndi(params);');
t = toc;
n = prod(n);
end
% ----------------------------------------------- %
function [t, n] = bench_nlTGV(gpuFlag)
magnitude = getfield(load('magn.mat'),'magn');
phase = getfield(load('phs_tissue.mat'),'phs_tissue');
mask = getfield(load('msk.mat'),'msk');
dr = getfield(load('spatial_res.mat'),'spatial_res');
n = size(mask);

kernel = dipole_kernel_fansi( n, dr, 0 );

TE = 25e-3;
B0 = 2.8936;
gyro = 2*pi*42.58;
phs_scale = TE * gyro * B0;
 
params = [];
params.input = phase * phs_scale;
params.weight = (magnitude .* mask);
params.K = kernel;
params.maxOuterIter = 50;
params.isGPU = gpuFlag;

params.alpha1 = 2e-4;

tic % use evalc to capture command window output of function and void
[~] = evalc('nlTGV(params);');
t = toc;
n = prod(n);
end
% ----------------------------------------------- %
function [t, n] = bench_compute_xsim(stream)
n = 512;
N = [n,n,n];

reset(stream,0);
img1 = randn(stream,N);
img2 = randn(stream,N);
tic
[~] = compute_xsim(img1, img2);
t = toc;
end
% ----------------------------------------------- %
function t = bench_imagesc3d2(isVisible)
chi = getfield(load('chi_cosmos.mat'),'chi_cosmos');
n = size(chi);

hh = figure('WindowStyle','normal');
set(hh,'pos','default','menubar','none','NumberTitle','off', ...
    'Name','imagesc3d2 benchmark','Visible', isVisible);
ScreenSize = get(0,'ScreenSize');
set(hh,'Position',[ScreenSize(3:4)*0.2,ScreenSize(3:4)*0.6]);

tic
for i = 1:min(n)
    imagesc3d2( chi, ones(3,1)*i, hh, [90,90,-90], [-0.1,0.1])
end
t = toc;
pause(2)
close(hh);
end
% ----------------------------------------------- %
function logBenchData(times,isGPU)

% Check for exising log file
if exist('benchmark_fansi.dat','file') ~= 2
    warning('No benchmark data file found, creating.')
    fp = fopen('benchmark_fansi.dat', 'a');
    fprintf(fp,'FANSI Benchmark Data.\n\n');
    fprintf(fp,'%75s%10s%10s%10s%10s%10s\n',...
                ' ','TKD','NDI','nlTGV','compute_xsim','imagesc3d2');
    fclose(fp);
end

% first collect computer information
if ismac
    status = zeros(7,1);
    [status(1), CpuStr] = system('sysctl -n machdep.cpu.brand_string');
    [status(2), CpuSpeed] = system('sysctl -n hw.cpufrequency');
    [status(3), noCores] = system('sysctl -n machdep.cpu.core_count');
    [status(4), compname] = system('hostname');
    [status(5), ostype] = system('sysctl -n kern.ostype');
    [status(6), osrelease] = system('sysctl -n kern.osrelease');
    [status(7), meminfo] = system('sysctl -n hw.memsize');
    
    status = sum(status);
elseif isunix
    status = zeros(3,1);
    % BUG: Adds extra linebreaks if cmd window too small, no easy fix as of
    % 2021/01:
    [status(1), sysinfo] = system('lscpu'); 
    [status(2), osinfo] = system('cat /etc/*-release');
    [status(3), meminfo] = system('free -g | grep Mem'); % in Gb
    [status(4), compname] = system('hostname');
    
    status = sum(status);
elseif ispc
    [status, sysinfo] = system('systeminfo');
    [~, sV] = memory;
else
    warning('Computer information not obtainable.')
    sysinfo = computer;
end

if status
    warning('Failure during system call for computer information.')
    sysinfo = computer;
end

if ismac
    ModelStr = sprintf('%s @ %1.4gGHz ', deblank(CpuStr), str2double(CpuSpeed)/1e9 );
    osStr = sprintf('%s %s',deblank(ostype),deblank(osrelease));
    memsize = round(str2double(meminfo)/1073741824);
    noCores = str2double(noCores);
    
elseif isunix
    ModelStr = regexp(sysinfo,'(?<=Model name:\s+)\w[^\n]*','match');
    ModelStr = ModelStr{:};
    noCores  = regexp(sysinfo,'(?<=CPU.{3}:\s+)\w[^\n]*','match','once');
    noCores = str2double(noCores);
    
    osinfo = regexprep(osinfo,'"','');
    osStr = regexp(osinfo,'(?<=PRETTY_NAME=)\w[^\n]*','match');
    osStr = osStr{:};

    memsize = str2double(regexp(meminfo, '[0-9]*', 'match'));
    memsize = memsize(1); % first entry of "free -g" is total memory
elseif ispc
    ModelStr = regexp(sysinfo,'(?<=\[01\]:\s)\w[^\n]*','match','once');
    [~,noCores] = evalc('feature(''numcores'')');
    
    osStr = regexp(sysinfo,'(?<=OS Name:\s+)\w[^\n]*','match');
    osStr = osStr{:};

    compname = regexp(sysinfo,'(?<=System Model:\s+)\w[^\n]*','match');
    compname = compname{:};
    
    memsize = round(sV.PhysicalMemory.Total/1073741824); % in Gb
end

% Get GPU info (if available)
if isGPU
    gpuStr = gpuDevice;
    gpuStr = [' ',gpuStr.Name,', '];
else
    gpuStr = '';
end

switch lower(computer)
    case ('pcwin64')
        osShort = 'Windows(R)';
    case ('glnxa64')
        osShort = 'Linux';
    case ('maci64')
        osShort = 'macOS';
    otherwise
        osShort = 'undetermined';
end


% Check for previous runs
logs = fileread('benchmark_fansi.dat');
if regexpi(logs,deblank(compname))
    reply = input(['There already appears to be benchmark data from ',...
        'this machine, would you like to add a new entry [y/N]?'],'s');
    if isempty(reply)
        reply = 'N';
    end
    if strcmpi(reply,'n')
        return
    end
end

specs = sprintf('%s, %s, %s',deblank(compname),osShort,ModelStr);
details = sprintf('\t%i cores, %s, %s, %i GB RAM,%s MATLAB %s\n',...
                            noCores,osStr,ModelStr,memsize,gpuStr,version);
    
% Write data to file
fp = fopen('benchmark_fansi.dat', 'a');

for time = times'
    timings = sprintf('%10.5f%10.5f%10.5f%10.5f%10.5f\n',time);
    fprintf(fp,'%-75s%s',specs,timings);
    fprintf(fp,details);
end

% Close the data file
fclose(fp);
end
