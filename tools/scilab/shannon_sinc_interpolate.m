

function [out] = shannon_sinc_interpolate(in, f)

l = size(in,2);
I = l * f;

% integration limit low
ill = -10;

% integration limit high
ilh = 10;

% interpolation start index
isi = -ill * f + f;

% interpolation end index
iei = I - (ilh * f);

% allocate array
out(I) = 0;

T = 1/50781.25;

% interpolate
for t = isi:iei
    for n = ill:ilh;
        mn = n + floor(t / f);
        mn = n + 11;
        out(t) = out(t) + in(mn) * sinc((t - mn * T)/T);
    end
end
