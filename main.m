%% Data Read in

nchannels_sniff = 8;
nchannels_ephys = 16;
ch_sniff = 8;

sniff_file = "\\F-moving-data\shnk3 (a)\080321_4131_session3_ADC.bin";
ephys_file = "\\F-moving-data\shnk3 (a)\080321_4131_session3_Ephys.bin";

adcx = LoadSniff(sniff_file);
size(adcx)
sniff = getSniff(adcx, nchannels_sniff, ch_sniff);
size(sniff)

%ephysx = reshapeEphys(LoadEphys(ephys_file), nchannels_ephys);

fprintf('Data Read-in Complete\n')

figure;
plot(sniff(1:300000))


%% Data Preprocessing

%sniff= removeJumps(sniff);

sniff = Resample_sniff(sniff);

figure
plot(sniff(1:10000))

%%
sniff_smooth = smooth(sniff, 25, 'sgolay');

figure;
plot(sniff_smooth(1:10000))

[pks, locs] = findpeaks(sniff_smooth, 'MinPeakProminence', 50);

figure;
hold on
plot(sniff)
scatter(locs, pks)
hold off


%%
ephysx_rs = resampleEphys(ephysx, nchannels_ephys);

fprintf('Preprocessing Complete\n')


%% Lock Ephys to inhalation

window_size = 1000;
nsniffs = 512;
begining = 9000;

sniff_activity = LockSniff(locs, ephysx_rs, window_size, nsniffs, begining, nchannels_ephys);

fprintf('LockSniff Complete\n')


%% Sortting

sorted_activity = frequencySort(sniff_activity, nchannels_ephys, nsniffs, window_size);


%% Plotting

for ii = 1:nchannels_ephys
    subplot(4,4,ii);

    imagesc(sorted_activity(:,:,ii));

    title(['Channel ' num2str(ii)]);
    axis square;
    

    x_lim = xlim;
    x_middle = mean(x_lim);
    xticks([x_middle - 500, x_middle, x_middle + 500]);
    
    yticklabels([])
    yticks([])
    xticklabels({'-500', '0', '500'});
end