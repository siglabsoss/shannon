% shannon_phase_map
% Converts a PN code to a phase map with a gaussian filter
% @param prnc pseudo random number code
% @param os oversample rate
% @param bt gaussian filter Bt
% @param cps chips per second
% @param ferror frequency offset error
% @param chip error in radians

function out = shannon_gen_pn( prnc, os, bt, cps, ferror, cerror )

l = size(prnc,2);
L = l * os;


phase(L) = 0;
phase(1) = 0;

% exdend prnc vector to allow for phase error
prnc(end+1) = phase(end);

for n = 2:L
    phase(n) = phase(n-1) + prnc(floor(l*(n)/L + cerror/(2*pi))+1) * pi / 2 / os;
end


% x-axis in units of seconds
xt = [0:1/cps/os:l/cps];
xt = xt(1:end-1);

% x-axis in units of chips
xc = [0:1/os:l];
xc = xc(1:end-1);

% create filter
gaussFilter = gaussfir(bt, 1, os);
phaseFiltered = conv(phase, gaussFilter, 'same');

out = 1i*sin(ferror*xt*2*pi + phaseFiltered) + cos(ferror*xt*2*pi + phaseFiltered);

