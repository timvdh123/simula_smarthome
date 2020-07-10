function sensor_labels2csv(s,fn)
FID = fopen(fn,'w');
fprintf(FID,'label,,id\n');
for i = 1:length(s.sensor_labels)
    fprintf(FID, '%s,,%d\n',s.sensor_labels{i, 2}, s.sensor_labels{i, 1});  
end
fclose(FID);
