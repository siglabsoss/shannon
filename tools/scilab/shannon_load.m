function out = shannon_load(fname, tbin, fbin)

s = dir(fname);

len = s.bytes / 8;

frames = floor(len / ( tbin * fbin ));

len = frames * tbin * fbin;

out(len) = 1i;

fid = fopen(fname);

out = fread(fid, len, 'single', 4);

fseek(fid, 4, 'bof');

out = out + 1i * fread(fid, len, 'single', 4);

fclose(fid);


out = reshape(out, [tbin fbin frames]);
