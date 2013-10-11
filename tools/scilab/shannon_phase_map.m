% shannon_phase_map
% Converts a PN code to a phase map with a gaussian filter
% @param prnc pseudo random number code
% @param os oversample rate
% @param bt gaussian filter Bt
% @param cps chips per second
% @param ferror frequency offset error
% @param chip error in radians

function out = shannon_phase_map( prnc, os, bt, cps, ferror, cerror )

close all

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

% modulus and zero for plot
%phase = mod(phase, 2 * pi) - pi;


subplot(2,2,[1 2]);
mtit('Quasi-coherent Receiver Analysis', 'yoff', 0.05, 'xoff', 0);

hold on

plot(xc, [phase; phaseFiltered]');

hold off

grid on

title('GMSK Phase');
legend show;
legend('Unfiltered',strcat('Filtered(bt=',num2str(bt),')'));

%set(findobj(gca,'Type','line'), 'LineWidth', 2, 'Color', 'r', ...
%    'Marker', 'o', 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'none', ...
%    'MarkerSize', 5);

% configure axis #1
xlabel('chip');


% configure Y Axis
% round to the nearest half pi
ymin = roundn(min(phase)/pi*2,0)*pi/2;
ymax = roundn(max(phase)/pi*2,0)*pi/2;
ylim([ymin,ymax]);
ylabel('phase');
yt = [ymin:pi/2:ymax];
set(gca, 'YTick', yt);
for n=1:size(yt,2);
    ytnames{n} = strcat(char(simplify(sym(yt(n)/pi))), 'pi');
end
set(gca, 'YTickLabel', ytnames);

% configure X axis
set(gca, 'XTick', [0:l]);
xlim([0,l]);


% configure axis #2
%ax2 = LinkTopAxisData([0:l], [0:1/cps:l/cps]*1e6, 'time(\mus)');

outRef = 1i*sin(phaseFiltered) + cos(phaseFiltered);
baseband = 0;
outRealRef = sin(baseband*xt*2*pi + phaseFiltered);
outReal = sin((baseband+ferror)*xt*2*pi + phaseFiltered);
out = 1i*sin(ferror*xt*2*pi + phaseFiltered) + cos(ferror*xt*2*pi + phaseFiltered);
%s = sin(freq*xt*2*pi + phase);
subplot(2,2,3);
%plot(xc, [real(out); imag(out); real(outRef); imag(outRef)]');
plot(xc, [outRealRef; outReal]');
title(strcat('Incoherence (Baseband = ', num2str(baseband), 'Hz, f_{error} = ', num2str(ferror), 'Hz)'));
legend show;
legend('Reference',strcat('f_{error}=',num2str(ferror),'Hz'));
%plot(xc, s);
xlabel('chip');
% configure X axis
set(gca, 'XTick', [0:l]);
xlim([0,l]);

subplot(2,2,4);

refCorr = xcorr(outRef, outRef) / os;
errCorr = xcorr(outRef, out) / os;
semilogy([abs(refCorr); abs(errCorr)]');
title('Cross-correlation');
legend show;
legend('Reference',strcat('f_{error}=',num2str(ferror),'Hz'));
%set(gca, 'YTick', [0 10 100 100]);
ylim([1 100]);
xlim([0 L*2]);


%ax3 = LinkTopAxisData([0:l], [0:1/cps:l/cps]*1e6, 'time(\mus)');
grid on

