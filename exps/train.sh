# This is a sample script that trains the Gaussian mixture representations for multi-sense embeddings on a small text corpus text8. 
# mkdir modelfiles
./multift skipgram -input "data/text8" -output modelfiles/text8  -lr 1e-2 -dim 300 \
    -ws 5 -epoch 30 -minCount 5 -loss ns -bucket 2000000 \
    -minn 3 -maxn 6 -thread 7 -t 1e-5 -lrUpdateRate 100 -multi 0 -var 1 -var_scale 5e-2 -margin 1 -notlog 0 -c 0.0 -min_logvar 0.01 -max_logvar 2.0
