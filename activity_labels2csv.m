function activity_labels2csv(s,fn)
FID = fopen(fn,'w');
fprintf(FID,'label,,id\n');
for i = 1:length(s.activity_labels)
    fprintf(FID, '%s,,%d\n',s.activity_labels{i}, i);  
end
fclose(FID);
