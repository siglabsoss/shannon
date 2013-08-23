
function out = shannon_demodulate(in, pn)

%clear all
close all

% demodulator constants (change these if needed)
nsc = size(pn,2); % number of spreading codes
osr = 1; % oversample rate
tx_bt = 0.5;
tx_chip_rate = 50781.25;
fmin = -1000; % minimum frequency (Hz)
fmax = 1000; % maximum frequency (Hz)
fsteps = 2000; % number of frequency scan steps

% calculated constants (don't change these)
f = linspace(fmin,fmax,fsteps);
p = linspace(0,nsc/tx_chip_rate*1000,nsc*osr);

% preallocate memory
cfc(fsteps, osr * nsc) = 0; % convolution filter coefficients
out(fsteps, osr * nsc) = 0; % output array

% generate PRN spreading code
%pn = sign(rand(nsc,1)-0.5)';

% generate reference waveforms for all frequencies
for n = 1:fsteps
    ref = shannon_gen_pn( pn, osr, tx_bt, tx_chip_rate, f(n), 0 );
    cfc(n,:) = fft( conj( fliplr( ref ) ) );
end

% generate (or sample) waveform to find
%in = shannon_gen_pn( pn, osr, tx_bt, tx_chip_rate, 120, 0 );
in = circshift(in', 100)';

% take the fft of input waveform
in_fft = fft(in);

% multidimensional convolution
for n = 1:fsteps
    out(n,:) = ifft( in_fft .* cfc(n,:) );
end

surf(p, f, abs(out),'EdgeColor','none');
xlabel('phase(ms)');
ylabel('frequency');
