I=imread('cameraman.tif');
[yy,xx]=meshgrid(1:256,1:256);
m1=( ((xx-100).^2+(yy-50).^2)<500 );
m2=( ((xx-128).^2+(yy-170).^2)<5000 );

data=I;
save in.mat data -v7.3
data=m1;
save m1.mat data -v7.3
data=m2;
save m2.mat data -v7.3

[a,ovl]=nxcorr(I,ones(size(I)),I,ones(size(I)));
data=a;
save ncc.mat data -v7.3
data=ovl;
save ncc_overlap.mat data -v7.3

[a,ovl]=nxcorr(I,m1,I,m2);
data=a;
save masked_ncc.mat data -v7.3
data=ovl;
save masked_ncc_overlap.mat data -v7.3




























