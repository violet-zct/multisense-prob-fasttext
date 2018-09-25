# This is a sample script that trains the Gaussian mixture representations for multi-sense embeddings on a small text corpus text8. 
# mkdir modelfiles
./multift skipgram -input "data/text8" -output modelfiles/lr0001var005  -lr 1e-3 -dim 50 \
    -ws 10 -epoch 10 -minCount 5 -loss ns -bucket 2000000 \
    -minn 3 -maxn 6 -thread 8 -t 1e-5 -lrUpdateRate 100 -multi 0 -var 1 -var_scale 5e-2 -margin 1 -notlog 0
