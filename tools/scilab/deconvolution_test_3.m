%% Load Raw Data
% This is slow -- Run by section. 

FILE_RF = '130731_4_rf.raw';
FILE_PN = '130731_2_pn.raw';
FILE_DE = '130731_1_decon.raw';

rf = shannon_convert(FILE_RF);
pn = shannon_convert(FILE_PN);
de = shannon_convert(FILE_DE);

%% Plot Raw Data
close all

figure(1)
plot([real(rf), imag(rf)])
title('Raw RF Data');

figure(2)
plot([real(pn), imag(pn)])
title('Raw PN Data');

figure(3)
plot([real(de), imag(de)])
title('Raw Deconvoluted Data');

%% Try deconvoluting the most trivial case (autocorrelation)
% This should output super nice triangular waveforms as the convolution
% window slides over the RF pn code
%
% Here, we are just coyping out a section of the RF waveform and doing an
% autocorrelation - ideal case. 

rf_pn = rf(81300:81300+800);
corr1 = xcorr(rf, rf_pn);

figure(4)
plot([real(corr1),imag(corr1)])
title('Trivial Case - Xcorr');


%% Try deconvoluting with our generated PN

gen_pn = pn(1000:1800-1);
corr2 = xcorr(rf, gen_pn);

figure(5)
plot([real(corr2),imag(corr2)])
title('Unmodified data - Xcorr');

%% Try frequency shifting
close all

% @10000 bins -61 seems optimal
% @1024 bins -5 seems optimal

% Do the fft 
NFFT = 1024;
x = linspace(1,800,800);
fft_1 = fft(gen_pn,NFFT);

%shift up the freq
ns = -6; %play with this

fft_shift = fft_1;
%fft_shift(800-ns:800) = 0;
fft_shift = circshift(fft_shift,[1 ns]);

%ifft 
ifft_1 = ifft(fft_shift);
ifft_2 = ifft_1(1:800);

% Try deconvoluting with our shifted PN
corr3 = xcorr(rf, ifft_2);
max(real(corr3)) %maximize this

figure(7)
plot([real(corr3); imag(corr3)]')
title('Shifted data - Xcorr');