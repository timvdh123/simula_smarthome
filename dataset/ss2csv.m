function ss2csv(ss,fn)
FID = fopen(fn,'w');
fprintf(FID,'start_time,end_time,id,val\n');
for i = 1:length(ss.d)
    fprintf(FID, '%s,%s,%d,%d\n',datestr(ss(i).start),datestr(ss(i).end),ss(i).id, ss(i).val);  
end
fclose(FID);
end
