
files=dir('*.mat');
lunghezza=length(files);
for i=1:lunghezza
    load(files(i).name);
    name = erase(files(i).name,".mat");
    name=strcat(name,'.png')
    val = (val - min(val) )/(max(val) - min(val));
    if length(val) >= 3600
        spectrogram(val(1:3600),hanning(32),'yaxis',10,1024,360);
        colorbar('off');
        axis off
    else
        spectrogram(val,hanning(32),'yaxis',10,1024,360);
        colorbar('off');
        axis off
    end
    saveas(gcf,name);
end
