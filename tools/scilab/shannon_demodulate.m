% shannon_demodulate
%
% @param in input buffer (double buffered)
% @param cfc correlation function coefficients
% @param osr oversample rate
%
% Both the input butter and cfc must be double buffered to prevent
% reflections in adjacent buffers. Output peaks correlate to END
% of spreading code.
%
function out = shannon_demodulate(in, cfc, fsteps)

close all

% calculated constants (don't change these)
len = size(cfc,2); % double buffer size

% take the fft of input waveform
in_fft = fft(in);

% preallocate output array for speed
out(fsteps,len) = 0;

% parametrized convolution - Iterate over multiple frequencies.
for n = 1:fsteps
    out(n,:) = ifft( in_fft .* circshift(cfc',n-fsteps/2)');
end

% circular shift to line up peaks with END of spreading code
out = circshift( out', len/4 )';

% remove padding
out = out( :,1:len/2 );

% find peaks (TODO)


% perform sinc interpolation on peaks to extract time-of-arrival (TODO)


