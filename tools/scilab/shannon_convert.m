%
% This function converts std::complex<float> to a matlab array
%
function [out] = shannon_convert(file_name)

s = dir(file_name);

final_size = s.bytes / 8;

fp = fopen(file_name);

out = complex(ones(1,final_size), ones(1,final_size));

for i = 1:length(out(:))
    out(i) = complex(fread(fp,1,'*single'),fread(fp,1,'*single'));
end

fclose(fp);
