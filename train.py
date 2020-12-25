import argparse
from datetime import datetime
from tensorboardX import SummaryWriter # install tensorboardX (pip install tensorboardX) before importing this package

def train(args, model=None, pad = 0):
    # LOG #
    fh = logging.FileHandler(f"./output/logs.txt")
                                      # create file handler which logs even debug messages
    logger.addHandler(fh)# add the handlers to the logger
    
    timestamp = datetime.now().strftime('%Y%m%d%H%M')
    tb_writer = SummaryWriter(f"./output/logs/{timestamp}") if args.visual else None

    # Set the random seed manually for reproducibility.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(device)


    config=get_config()

    if args.visual:
        json.dump(config, open(f'./output/config_{timestamp}.json', 'w'))# save configs

    ###############################################################################
    # Load data
    ###############################################################################
    data_path = args.data_path+args.dataset+'/'
    train_set = DialogDataset(os.path.join(data_path, 'train.h5'), config['diaglen'], config['maxlen'])
    valid_set = DialogDataset(os.path.join(data_path, 'valid.h5'), config['diaglen'], config['maxlen'])
    test_set = DialogDataset(os.path.join(data_path, 'test.h5'), config['diaglen'], config['maxlen'])
    vocab = load_dict(os.path.join(data_path, 'vocab.json'))
    ivocab = {v: k for k, v in vocab.items()}
    n_tokens = len(ivocab)
    metrics=Metrics()    
    print("Loaded data!")

    ###############################################################################
    # Define the models
    ###############################################################################
    if model is None:
        model = MyModel(config, n_tokens)

    if args.reload_from>=0:
        load_model(model, args.reload_from)
        
    model=model.to(device)

    logger.info("Training...")
    best_perf = -1
    itr_global=1
    start_epoch=1 if args.reload_from==-1 else args.reload_from+1
    for epoch in range(start_epoch, config['epochs']+1):
        epoch_start_time = time.time()
        itr_start_time = time.time()
        
        # shuffle (re-define) data between epochs   
        train_loader=torch.utils.data.DataLoader(dataset=train_set, batch_size=config['batch_size'],
                                                 shuffle=True, num_workers=1, drop_last=True)
        n_iters=train_loader.__len__()
        itr = 1
        for batch in train_loader:# loop through all batches in training data
            model.train()
            context, context_lens, utt_lens, floors, response, res_lens = batch

 #           max_ctx_len = max(context_lens)
            max_ctx_len = context.size(1)
            context, utt_lens = context[:,:max_ctx_len,1:], utt_lens[:,:max_ctx_len]-1
                                    # remove empty utterances in context
                                    # remove the sos token in the context and reduce the context length     
#################################################
            utt_lens[utt_lens<=0]=1
#################################################
            batch_gpu = [tensor.to(device) for tensor in [context, context_lens, utt_lens, response, res_lens]] 
            train_results = model.train_batch(*batch_gpu)
                     
            if itr % args.log_every == 0:
                elapsed = time.time() - itr_start_time
                log = '%s|%s@gpu%d epo:[%d/%d] iter:[%d/%d] step_time:%ds elapsed:%s'\
                %(args.model, args.dataset, args.gpu_id, epoch, config['epochs'],
                         itr, n_iters, elapsed, timeSince(epoch_start_time,itr/n_iters))
                logger.info(log)
                logger.info(train_results)
                if args.visual:
                    tb_writer.add_scalar('train_loss', train_results['train_loss'], itr_global)

                itr_start_time = time.time()    
                
            if itr % args.valid_every == 0 and False:
                logger.info('Validation ')
                valid_loader=torch.utils.data.DataLoader(dataset=valid_set, batch_size=config['batch_size'], shuffle=True, num_workers=1)
                model.eval()    
                valid_losses = []
                for context, context_lens, utt_lens, floors, response, res_lens in valid_loader:
 #                   max_ctx_len = max(context_lens)
                    max_ctx_len = context.size(1)
                    context, utt_lens = context[:,:max_ctx_len,1:], utt_lens[:,:max_ctx_len]-1
                             # remove empty utterances in context
                             # remove the sos token in the context and reduce the context length
#################################################
                    utt_lens[utt_lens<=0]=1
#################################################
                    batch = [tensor.to(device) for tensor in [context, context_lens, utt_lens, response, res_lens]]
                    valid_results = model.valid(*batch)    
                    valid_losses.append(valid_results['valid_loss'])
                if args.visual: tb_writer.add_scalar('valid_loss', np.mean(valid_losses), itr_global)
                logger.info({'valid_loss':np.mean(valid_losses)})    
                
            itr += 1
            itr_global+=1            
            
            if itr_global % args.eval_every == 0:  # evaluate the model in the validation set
                model.eval()          
                logger.info("Evaluating in the validation set..")

                valid_loader=torch.utils.data.DataLoader(dataset=valid_set, batch_size=1, shuffle=False, num_workers=1)

                f_eval = open(f"./output/tmp_results/iter{itr_global}.txt", "w")
                repeat = 10            
                eval_results = evaluate(model, metrics, valid_loader, vocab, repeat, f_eval)
                bleu = eval_results['recall_bleu']
                if bleu> best_perf:
                    save_model(model, 0)#itr_global) # save model after each epoch
                if args.visual:
                    tb_writer.add_scalar('recall_bleu', bleu, itr_global)
                
        # end of epoch ----------------------------
               # model.adjust_lr()

    return model