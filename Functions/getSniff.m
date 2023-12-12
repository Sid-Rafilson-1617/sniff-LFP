function sniff = getSniff(adcx, nchannels, ch)
    % Reshape binary vector to N channel matrix

    adcx = reshape(adcx, [], 1);
    adcx = reshape(adcx,[nchannels, round(length(adcx)/nchannels)]);
    sniff = adcx(ch, :);

end