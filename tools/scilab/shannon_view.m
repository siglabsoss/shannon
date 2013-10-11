
function shannon_view(cps, fbins)

clear dump
clear dump_reshape

%cps = 256;

dump = shannon_load('Z:\popwi_joel\dump.raw', cps * 2, fbins);

frames = size(dump, 3);
len = size(dump, 1) * frames / 2;

dump_reshape(len,fbins) = 1i;

% demodulate (piecewise)
%for n = 2:N
%    out(:,(n-1)*osl+1:n*osl) = ...
%        shannon_demodulate(in((n-2)*osl+1:n*osl), cfc, fsteps);
%end
for n = 1:frames
    a = circshift(dump(:,:,n), cps/4);
    dump_reshape(1+(n-1)*cps:n*cps,:) = a(1:cps,:);
end

surf(abs(dump_reshape(:,1:end)), 'EdgeColor', 'none');
title(strcat(num2str(cps), ' chips'));


