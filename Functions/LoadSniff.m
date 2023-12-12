function adcx = LoadSniff(sniff_file)

    adc = fopen(sniff_file, 'r');
    adcx = fread(adc, 'uint16');
    fclose(adc);

end