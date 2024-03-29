clear all
close all

% example constants (change these if needed)
osr = 1; % oversample rate
phase = 0; % synthesized input phase offset
toa = 0; % synthesized time of arrival
bt = 0.5; % Gaussian filter -3dB value for GMSK transmitter
tx_chip_rate = 50781.25;
ferror = 0; % frequency error
phase = 0; % phase of pn code in radians
%pn = sign(rand(nsc,1)-0.5)';
pn = [-1, 1, 1,-1,-1, 1, 1, 1,-1, 1, 1, 1, 1,-1, 1,-1, 1, 1, 1,...
       1, 1,-1, 1,-1,-1,-1,-1, 1, 1, 1,-1,-1,-1, 1,-1, 1,-1,-1,...
       1,-1,-1,-1,-1,-1,-1, 1, 1, 1,-1, 1,-1, 1,-1, 1, 1,-1,-1,...
      -1,-1,-1,-1, 1, 1,-1,-1,-1,-1,-1, 1,-1,-1,-1,-1, 1,-1, 1,...
       1, 1,-1,-1, 1,-1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1, 1,-1,...
      -1, 1, 1, 1,-1, 1,-1,-1,-1, 1,-1, 1,-1, 1, 1, 1,-1, 1,-1,...
      -1,-1, 1,-1,-1,-1, 1, 1,-1, 1, 1, 1,-1, 1, 1,-1,-1,-1,-1,...
       1, 1, 1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1, 1, 1, 1,...
       1,-1,-1, 1, 1,-1, 1, 1,-1,-1,-1,-1,-1, 1,-1,-1, 1, 1,-1,...
      -1, 1, 1, 1, 1, 1,-1,-1, 1, 1,-1, 1,-1, 1,-1, 1,-1,-1, 1,...
       1, 1, 1, 1, 1,-1,-1,-1,-1, 1, 1,-1,-1, 1,-1, 1,-1,-1, 1,...
      -1,-1,-1, 1, 1,-1,-1,-1,-1, 1,-1,-1, 1,-1, 1, 1, 1,-1,-1,...
      -1,-1,-1,-1,-1,-1, 1,-1, 1, 1, 1, 1, 1,-1, 1, 1, 1,-1,-1,...
      -1, 1,-1, 1,-1, 1,-1,-1,-1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,...
      -1, 1, 1, 1,-1,-1,-1,-1, 1, 1,-1, 1, 1,-1, 1,-1, 1,-1, 1,...
       1, 1,-1,-1, 1,-1,-1, 1, 1,-1, 1,-1, 1, 1,-1, 1, 1, 1,-1,...
       1,-1, 1, 1, 1, 1, 1,-1,-1,-1,-1, 1, 1, 1, 1, 1, 1,-1,-1,...
       1, 1,-1,-1, 1,-1,-1,-1,-1, 1, 1,-1, 1,-1, 1,-1,-1, 1, 1,...
       1, 1, 1,-1,-1,-1,-1, 1, 1,-1, 1,-1,-1, 1, 1,-1, 1,-1,-1,...
       1, 1,-1,-1, 1,-1, 1, 1, 1,-1,-1, 1, 1,-1, 1, 1, 1, 1,-1,...
       1,-1, 1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1, 1, 1, 1,-1,-1,-1,...
      -1, 1,-1,-1, 1,-1,-1, 1,-1,-1, 1,-1, 1,-1,-1, 1,-1, 1, 1,...
      -1, 1, 1, 1,-1,-1,-1,-1, 1,-1,-1,-1,-1,-1, 1,-1,-1,-1, 1,...
       1, 1,-1,-1, 1, 1,-1, 1,-1,-1, 1, 1, 1, 1, 1, 1,-1, 1, 1,...
      -1, 1, 1, 1,-1, 1,-1,-1, 1, 1, 1,-1,-1, 1, 1,-1,-1,-1,-1,...
       1,-1,-1, 1, 1,-1,-1, 1,-1, 1, 1,-1,-1, 1, 1, 1,-1,-1, 1,...
       1, 1,-1, 1, 1, 1, 1, 1, 1,-1,-1,-1, 1,-1,-1, 1,-1, 1 ]; % __code_m4k

% calculated constants (don't change these)
nsc = size(pn,2); % number of spreading codes
dbs = nsc*osr*2; % double buffer size
sbs = nsc*osr;   % single buffer size
    
    
% generate (or sample) waveform to find
in = shannon_gen_pn( pn, bt );

% pad (double buffer)
in = padarray(in', sbs/2)';

% phase shift (test)
in = circshift(in', round(phase / 2 / pi * osr) + toa * osr)';

% generate convolution filter coefficients
cfc = shannon_calculate_cfc(pn, sbs, bt);

% demodulate
out = shannon_demodulate(in, cfc, 80) / osr;

% display
surf(abs(out),'EdgeColor','none');
zlim([0 500]);
title('GMSK spreading (512 chips)');
xlabel('phase(ms)');
ylabel('frequency');

