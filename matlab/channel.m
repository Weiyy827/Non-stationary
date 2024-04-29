clear

load("channel_coeff.mat")

fs = 1e10;

m_sequence = mseq([0 0 0 1 1 1 0 1]);
txSymbol = pskmod(m_sequence,2);
txWaveform = txSymbol;

clusters = [0,1,2];
for cluster = clusters
    coeff = NLOS_coeff{cluster+1};
    sample_delay(cluster+1) = coeff.delay * fs;

    filter_coeff = [zeros(1,round(sample_delay(cluster+1))),1];
    rxWaveform(cluster+1,:) = filter(filter_coeff,1,txWaveform);%把信号过时延
    rxWaveform(cluster+1,:) = rxWaveform(cluster+1,:) * coeff.coeff;%其他抽头与不同衰落相乘
end
rxWaveform = sum(rxWaveform,1);

corr = abs(xcorr(rxWaveform,txWaveform));
figure();
plot(corr)
