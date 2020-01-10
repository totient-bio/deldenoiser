#!/usr/bin/bash
deldenoiser --design ./input/size20x20x20_design.tsv  \
            --postselection_readcount ./input/size20x20x20_postselection_readcount.tsv  \
            --output_prefix ./output/size20x20x20  \
            --dispersion 1.0  \
            --regularization_strength 1.0  \
            --yields ./input/size20x20x20_yield.tsv  \
            --preselection_readcount ./input/size20x20x20_preselection_readcount.tsv
