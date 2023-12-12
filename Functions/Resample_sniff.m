function sniff = Resample_sniff(sniff)
    % resample to 1khz

    resample_factor = 30;
    sniff = resample(sniff, 1, resample_factor);

end