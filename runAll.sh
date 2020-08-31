## PASCAL
python main.py \
    -type pascal \
    -topK 3 \
    -theta 0.6 \
    -epochs 40 \
    -lr 0.00003 \
    -batch-size 128 \
    -test-batch-size 128 \
    -name pascalBaseline #\
    # -write \
    # -plot


# Running for theta 0.4(left pane) and 0.6 (right pane) on anandibai

# ## MSCOCO
# python main.py \
#     -type mscoco \
#     -topK 3 \
#     -theta 0.5 \
#     -epochs 50 \
#     -lr 0.00003 \
#     -batch-size 128 \
#     -test-batch-size 128 \
#     -name mscocoBaseline #\
#     # -write \
#     # -plot

## NUSWIDE
# python main.py \
#     -type nuswide \
#     -topK 3 \
#     -theta 0.5 \
#     -epochs 40 \
#     -lr 0.00003 \
#     -batch-size 128 \
#     -test-batch-size 128 \
#     -name nuswideBaseline #\
    # -write \
    # -plot