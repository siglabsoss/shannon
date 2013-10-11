% shannon_calculate_cfc
% Calculates Convolution Filter Coefficients for a given
% Pseudo Random Number (PRN) spreading code.
%
% Code is padded to prevent mirroring.
% PN code is in the format [-1, 1,-1, 1, 1, .......]
% osl - oversampled symbol length (in samples)

function cfc = shannon_calculate_cfc(pn, osl, bt)

% demodulator constants (change these if needed)
ncs = size(pn,2); % number of chips per symbol
tx_chip_rate = 50781.25;

% generate and pad matched filter
ref = shannon_gen_pn(pn, bt)';

% interpolate filter to sample frequency (verified JDB: 8/24/2013)
bts = (1:ncs)'; % bandlimited time series
its  = linspace(1, ncs, osl)'; % interpolated time series
ref = sinc(its(:,ones(ncs,1)) - bts(:,ones(osl,1))')*ref; 

% pad array 
ref = padarray(ref, osl/2);
   
% calculate convolution filter coefficients
cfc = fft( conj( fliplr( ref' ) ) );
