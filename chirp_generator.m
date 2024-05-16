%%
clear;clc;
Tc = 15.2*(10^-6);
fs = 2*(10^9);
t = 0:1/fs:Tc-1/fs;

% band-width
Bmin = 400*(10^6);
Bmax = 800*(10^6);

bits_Bandwidth = 7;
L = 2^bits_Bandwidth;
Bl = Bmin : (Bmax-Bmin)/L : ((L-1)*Bmax+Bmin)/L;

% frequency
fmin = -100*(10^6);
fmax = 100*(10^6);

bits_frequency = 6;
M = 2^bits_frequency;
fm = fmin : (fmax-fmin)/M : ((M-1)*fmax+fmin)/M;

% chirps

n = 256;    % we need 256 samples
Bsample = randsample(Bl,n,true);
fsample = randsample(fm,n,true);
sample = chirp_sample(t,Bsample,fsample,n,Tc);
%%
SNR = 10;
sample_10 = AWGN(sample,SNR);
%%
SNR = 20;
sample_20 = AWGN(sample,SNR);
%save('train_sample.mat',"sample_20")
%%
SNR = 0;
sample_0 = AWGN(sample,SNR);
%%
save('samples.mat',"sample_0","sample_20","sample_10")
save('indices.mat',"fsample","Bsample")

function sample = chirp_sample(duration,B,f,n,Tc)

    syms t a b
    sample0 = [];
    x = exp( 2*pi*1i*(0.5*a*t^2 + b*t) );

    for i = 1:n
        ch1 = subs(x,a,B(i)/Tc);
        ch2 = subs(ch1,b,f(i)-B(i)/2);
        ch = double(subs(ch2,t,duration));
        sample0 = [sample0,ch,zeros(1,200)];
    end
    sample = sample0;
end

function noisy_sample = AWGN(sample,SNR)
    [r,lags] = xcorr(sample,'biased');     % autocorrelation
    i = find(lags==0);      % zero lag element is the power of signal
    pow = 10*log10(r(i));    
    noisy_sample = awgn(sample,SNR,pow); 
end