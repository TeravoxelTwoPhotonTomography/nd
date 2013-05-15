function [z,overlap]=nxcorr(a,ma,b,mb)
% [z,overlap]=nxcorr(a,ma,b,mb)
%
% proof of concept masked normalized cross correlation
%
% 4 "register" arrays
% 2 output array (out-of-place)
% 3 forward fft's, 3 ifft's
%
% This slower (for small inputs?) that Dirk Padfiled's matlab
% implementation, but when implemented in C/CUDA proves that I can do it
% with relatively small memory overhead.
%
% The reflect function here is what is slowing everything down.
% This implementation should be n-dimensional, but I haven't verified that.
%
% Nathan Clack, May 2013
form=@(a,b)  double(a)+1i.*double(b);
first=@(a)   0.5.*(a+conj(reflect(a)));
second=@(a) -0.5i.*(a-conj(reflect(a)));

a(ma==0)=0; % apply mask
b(mb==0)=0;

sza=size(a);
szb=size(b);
a =padarray(double(a) ,szb,0,'post');
ma=padarray(double(ma),szb,0,'post');
b =padarray(double(b) ,sza,0,'post');
mb=padarray(double(mb),sza,0,'post');

r1=fftn(form(a,b));
r2=fftn(form(ma,mb));
r3=ifftn(form(...
  first(r1).*conj(second(r1)),...
  first(r2).*conj(second(r2)))); %ifft(a.b*),ifft(ma.mb*)
r4=ifftn(form(...
  first(r1).*conj(second(r2)),...
  first(r2).*conj(second(r1)))); %ifft(a.mb*), ifft(ma.b*)
%clear r1; % don't need to clear, reuse

r1=fftn(form(a.*a,b.*b));
r1=ifftn(form(...
  first(r1).*conj(second(r2)),... %ifft(aa.mb*), ifft(ma.bb*)
  first(r2).*conj(second(r1))));
clear r2;

overlap=max(round(imag(r3)),1e-3);
z=real(r3)-real(r4).*imag(r4)./overlap;
z=z./max(real(sqrt((real(r1)-(real(r4).^2)./overlap).*...
        (imag(r1)-(imag(r4).^2)./overlap))),1e-3);

  function z=reflect(z)
    for i=1:length(size(z))
      z=shiftdim(flipdim(circshift(z,-1),1),1);
    end
  end
  
end
