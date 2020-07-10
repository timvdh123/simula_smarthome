function as2csv(as,fn)
FID = fopen(fn,'w');
fprintf(FID,'start_time,end_time,id\n');
for i = 1:length(as.d)
    fprintf(FID, '%s,%s,%d\n',datestr(as(i).start),datestr(as(i).end),as(i).id);  
end
fclose(FID);
