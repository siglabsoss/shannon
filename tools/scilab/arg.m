function out = arg(in)

out = atan(imag(in)./real(in)) + pi/2*sign(imag(in)).*(1-sign(real(in)));
