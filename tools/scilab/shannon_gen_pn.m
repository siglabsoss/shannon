% shannon_phase_map
% Converts a PN code to a phase map with a gaussian filter
% @param prnc pseudo random number code
% @param bt gaussian filter Bt


function out = shannon_gen_pn( prnc, bt )

% oversample output (if desired)
os = 1;

% chip phase error (if desired)
cerror = 0;

% baseband frequency error (if desired)
ferror = 0;
cps = 50781.25;

% sample size
l = size(prnc,2);
L = l * os;

% preallocate array
phase(L) = 0;

% exdend prnc vector to allow for phase error (if desired)
prnc(end+1) = phase(end);

% x-axis in units of seconds
xt = [0:1/cps/os:l/cps];
xt = xt(1:end-1);

for n = 2:L
    phase(n) = phase(n-1) + prnc(floor(l*(n)/L + cerror/(2*pi))+1) * pi / 2 / os;
end

% create filter
gaussFilter = gaussfir(bt, 1, os);
phaseFiltered = conv(phase, gaussFilter, 'same');

out = 1i*sin(ferror*xt*2*pi + phaseFiltered) + cos(ferror*xt*2*pi + phaseFiltered);

