N=64;

%%
a=single(1:N);
x=single(rand(size(a)));
b=single(rand(size(a)));

z=a.*x+b;
save a_f32.mat a -v7.3;
save x_f32.mat x -v7.3;
save b_f32.mat b -v7.3;
save z_f32.mat z -v7.3;

%%
a=uint16(a);
x=uint16(1000*x);
b=uint16(1000*b);
z=a.*x+b;
save a_u16.mat a -v7.3;
save x_u16.mat x -v7.3;
save b_u16.mat b -v7.3;
save z_u16.mat z -v7.3;