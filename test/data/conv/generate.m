%% Prep
[ii,jj,kk]=meshgrid(1:256,1:256,1:256);
r = sqrt( (ii-128).^2 + (jj-128).^2 + (kk-128).^2);
a = single(sin((2*pi/100000.0).*(r-50).*(r-100).*(r-150)));

avg=[1 1 1]/3.0; % box filter

%% Convolution on each dimension
save orig.mat a -v7.3
avg0=imfilter(a,avg','replicate');
avg1=imfilter(a,avg,'replicate');
avg2=imfilter(a,reshape(avg,[1 1 3]),'replicate');
save avg0.mat avg0 -v7.3
save avg1.mat avg1 -v7.3
save avg2.mat avg2 -v7.3

%% seperable convolution
avg1=imfilter(avg0,avg,'replicate');
avg2=imfilter(avg1,reshape(avg,[1 1 3]),'replicate');
save avg.mat avg -v7.3





