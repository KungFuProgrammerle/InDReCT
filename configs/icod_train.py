has_test = True
deterministic = True
use_custom_worker_init = True
log_interval = 200
base_seed = 112358

__BATCHSIZE = 4
__NUM_EPOCHS = 40
__NUM_TR_SAMPLES = 3040 + 1000
__ITER_PER_EPOCH = __NUM_TR_SAMPLES // __BATCHSIZE  # drop_last is True
__NUM_ITERS = __NUM_EPOCHS * __ITER_PER_EPOCH

train = dict(

    scales=(



        1.0,
        0.5,
        1.5,
        2.0,
        2.5







    ),#,1.2,1.4,1.6,1.8
    scales_split=(),#zoomnext lide num_zum xiugai 3.0,4.0
    num_split=3,
    batch_size=__BATCHSIZE,
    num_workers=2,
    use_amp=True,
    num_epochs=__NUM_EPOCHS,
    epoch_based=True,
    num_iters=None,
    lr=0.0001,
    grad_acc_step=1,
    optimizer=dict(
        mode="adam",
        set_to_none=False,
        group_mode="finetune",
        cfg=dict(
            weight_decay=0,
            diff_factor=0.1,
        ),
    ),
    sche_usebatch=True,
    scheduler=dict(
        warmup=dict(
            num_iters=0,
            initial_coef=0.01,
            mode="linear",
        ),
        mode="step",
        cfg=dict(
            milestones=int(__NUM_ITERS * 2 / 3),
            gamma=0.1,
        ),
    ),
    bn=dict(
        freeze_status=True,
        freeze_affine=True,
        freeze_encoder=False,
    ),
    data=dict(
        shape=dict(h=384, w=384),
        # names=["cod10k_tr","camo_tr"],
        names=["cod10k_tr"],
        addnames=[],

    ),
)

test = dict(
    batch_size=__BATCHSIZE,
    num_workers=2,
    clip_range=None,
    data=dict(
        shape=dict(h=384, w=384),
        names=[ 'camo_te',"cod10k_te", "nc4k","SEG", "MoCA",],
        # names=["cod10k_te",'camo_te'],

    ),
)
