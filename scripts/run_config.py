def create_run_name(args, training_cfg):
    '''
    create unique identical run name for each run
    :param args: from argparse
    :param training_cfg: from configuration file
    :return: unique identical run name
    '''
    run_name = ""

    if args.do_train:
        run_name += "train-"

    run_name += args.model

    train_bz = training_cfg.get('hyper').get('train_batch_size')
    val_bz = training_cfg.get('hyper').get('val_batch_size')
    lr = training_cfg.get('hyper').get('lr')
    run_name = run_name + f"-hyper-train{train_bz}val{val_bz}lr{lr}-"

    run_name += "-".join(args.data)

    run_name = run_name + "-prompt_" + str(args.input_format)

    run_name = run_name + "-" + str(args.seed)

    return run_name