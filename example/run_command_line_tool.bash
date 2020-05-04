#!/usr/bin/bash
deldenoiser --design ./input/size20x20x20_design.tsv  \
            --postselection_readcount ./input/size20x20x20_postselection_readcount.tsv  \
            --output_prefix ./output/size20x20x20  \
            --dispersion 1.0  \
            --regularization_strength 1.0  \
            --yields ./input/size20x20x20_yields.tsv  \
            --preselection_readcount ./input/size20x20x20_preselection_readcount.tsv \
            --maxiter 20 \
            --inner_maxiter 10 \
            --tolerance 0.01 \
            --parallel_processes 8 \
            --maxyield 0.95 \
            --minyield 1e-6 \
            --F_init 0.0 \
            --max_downsteps 10

gzip -df ./output/size20x20x20_fullcycleproducts.tsv.gz
gzip -df ./output/size20x20x20_truncates.tsv.gz
gzip -df ./output/size20x20x20_tag_imbalance_factors.tsv.gz
