
function [p, f, out] = shannon_demodulate(in, pn, osr)

close all

% demodulator constants (change these if needed)
nsc = size(pn,2); % number of spreading codes
tx_bt = 0.5; % Gaussian filter -3dB value for GMSK transmitter
tx_chip_rate = 50781.25;
fmin = -2000; % minimum frequency (Hz)
fmax = 2000; % maximum frequency (Hz)
fsteps = 1024; % number of frequency scan steps

% calculated constants (don't change these)
f = linspace(fmin,fmax,fsteps);
p = linspace(0,nsc/tx_chip_rate*1000,nsc*osr);
dbs = nsc * osr * 2; % double buffer size
sbs = nsc * osr; % single buffer size

% preallocate memory
ref(fsteps, dbs) = 0; % reference prn (padded)
cfc(fsteps, dbs) = 0; % convolution filter coefficients
pad(fsteps, dbs) = 0; % padded output array

% generate and pad matched filter for all possible frequencies
for n = 1:fsteps
    ref(n,sbs/2+1:sbs+sbs/2) = ...
        shannon_gen_pn( pn, osr, tx_bt, tx_chip_rate, f(n), 0 );
end

% calculate convolution filter coefficients
cfc = fft( conj( fliplr( ref' ) ) )';

% take the fft of input waveform
in_fft = fft(in);

% parametrized convolution (shift to line up with END of spreading code)
for n = 1:fsteps
    pad(n,:) = circshift( ifft( in_fft .* cfc(n,:))', sbs/2 )'/osr;
end

out = pad(:,1:sbs);
