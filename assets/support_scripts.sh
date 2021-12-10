# Grep results
grep "Average Precision  (AP) @\[ IoU=0.50      | area=   all | maxDets=100 \]" conf3/train_log.txt > conf3/result.txt

# Grep recall
rep "Average Recall     (AR) @\[ IoU=0.50:0.95 | area=   all | maxDets=100 \]" conf3/train_log.txt > conf3/result_recall.txt

