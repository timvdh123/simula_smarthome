function struct2csv(s, name)
    ss2csv(s.ss, sprintf('%s.ss.csv',name));
    as2csv(s.as, sprintf('%s.as.csv',name));
    activity_labels2csv(s, sprintf('%s.activity_labels.csv',name));
    sensor_labels2csv(s, sprintf('%s.sensor_labels.csv',name));
end