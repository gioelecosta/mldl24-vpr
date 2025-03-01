import os
import pytorch_lightning as pl
import torch
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim import lr_scheduler
import drive.MyDrive.Project.code.utils as utils
from pytorch_lightning.loggers import TensorBoardLogger
from drive.MyDrive.Project.code.dataloaders.GSVCitiesDataloader import GSVCitiesDataModule
from drive.MyDrive.Project.code.models import helper
import glob
import numpy as np
import matplotlib.pyplot as plt 

IMAGENET_MEAN_STD = {'mean': [0.485, 0.456, 0.406], 
                     'std': [0.229, 0.224, 0.225]}

class VPRModel(pl.LightningModule):
    """
    This is the main model for Visual Place Recognition
    we use Pytorch Lightning for modularity purposes.
    """

    def __init__(self,
                #---- Backbone
                backbone_arch='resnet18',
                pretrained=True,
                layers_to_freeze=1,
                layers_to_crop=[],
                
                #---- Aggregator
                agg_arch='ConvAP', # ConvAP, AVG, GeM, MixVPR
                agg_config={},
                
                #---- Train hyperparameters
                lr=0.03, 
                optimizer='adam', # AdamW
                weight_decay=0,
                momentum=0.9,
                warmpup_steps=500,
                milestones=[5, 10, 15], #list of epochos when the learning rate will be reduced (multiplied by lr_mult)
                lr_mult=0.3,
                
                #----- Loss
                loss_name='MultiSimilarityLoss', # 'MultiSimilarityLoss'
                miner_name='MultiSimilarityMiner', #to select a batch of most imformative samples, 'MultiSimilarityMiner'
                miner_margin=0.1,
                faiss_gpu=False #faiss, library for efficient searching os similarity
                 ):
        super().__init__()
        self.encoder_arch = backbone_arch
        self.pretrained = pretrained
        self.layers_to_freeze = layers_to_freeze
        self.layers_to_crop = layers_to_crop

        self.agg_arch = agg_arch
        self.agg_config = agg_config

        self.lr = lr
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.warmpup_steps = warmpup_steps
        self.milestones = milestones
        self.lr_mult = lr_mult

        self.loss_name = loss_name
        self.miner_name = miner_name
        self.miner_margin = miner_margin
        
        self.save_hyperparameters() # write hyperparams into a file
        
        self.loss_fn = utils.get_loss(loss_name)
        self.miner = utils.get_miner(miner_name, miner_margin)
        self.batch_acc = [] # we will keep track of the % of trivial pairs/triplets at the loss level 

        self.faiss_gpu = faiss_gpu
        
        # ----------------------------------
        # get the backbone and the aggregator
        self.backbone = helper.get_backbone(backbone_arch, pretrained, layers_to_freeze, layers_to_crop)
        self.aggregator = helper.get_aggregator(agg_arch, agg_config)
        
    # the forward pass of the lightning model
    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregator(x)
        return x
    
    # configure the optimizer 
    def configure_optimizers(self):
        if self.optimizer.lower() == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), 
                                        lr=self.lr, 
                                        weight_decay=self.weight_decay)
        elif self.optimizer.lower() == 'adam':
            optimizer = torch.optim.AdamW(self.parameters(), 
                                        lr=self.lr, 
                                        weight_decay=self.weight_decay)
        else:
            raise ValueError(f'Optimizer {self.optimizer} has not been added to "configure_optimizers()"')
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.lr_mult)
        #class of torch.optim for decreasing the learning rate multipling  lr for lr_mult each milestons epochs
        return [optimizer], [scheduler]
    
    # configure the optizer step, takes into account the warmup stage
    def optimizer_step(self,  epoch, batch_idx,
                        optimizer, optimizer_idx, optimizer_closure,
                        on_tpu, using_native_amp, using_lbfgs):
        # warm up lr
        if self.trainer.global_step < self.warmpup_steps:
            lr_scale = min(1., float(self.trainer.global_step + 1) / self.warmpup_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.lr
        optimizer.step(closure=optimizer_closure)
        
    #  The loss function call (this method will be called at each training iteration)
    def loss_function(self, descriptors, labels):
        # we mine the pairs/triplets if there is an online mining strategy
        if self.miner is not None:
            miner_outputs = self.miner(descriptors, labels)
            loss = self.loss_fn(descriptors, labels, miner_outputs)
            
            nb_samples = descriptors.shape[0]
            nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
            batch_acc = 1.0 - (nb_mined/nb_samples)

        else: # no online mining
            loss = self.loss_fn(descriptors, labels)
            batch_acc = 0.0
            if type(loss) == tuple: 
                loss, batch_acc = loss

        # keep accuracy of every batch and later reset it at epoch start
        self.batch_acc.append(batch_acc)
        # log it
        self.log('b_acc', sum(self.batch_acc) /
                len(self.batch_acc), prog_bar=True, logger=True)
        return loss
    
    # This is the training step that's executed at each iteration
    def training_step(self, batch, batch_idx):
        # batch is the output of data iterable, normally a DataLoader.
        # batch_idx is the index of this batch.

        places, labels = batch
        
        # Note that GSVCities yields places (each containing N images)
        # which means the dataloader will return a batch containing BS places
        BS, N, ch, h, w = places.shape
        
        # reshape places and labels
        images = places.view(BS*N, ch, h, w)
        labels = labels.view(-1)

        # Feed forward the batch to the model
        descriptors = self(images) # Here we are calling the method forward that we defined above
        loss = self.loss_function(descriptors, labels) # Call the loss_function we defined above
        
        self.log('loss', loss.item(), logger=True)
        return {'loss': loss}
    
    # This is called at the end of eatch training epoch
    def training_epoch_end(self, training_step_outputs):
        # we empty the batch_acc list for next epoch
        self.batch_acc = []

    # For validation, we will also iterate step by step over the validation set
    # this is the way Pytorch Lghtning is made. All about modularity, folks.
    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        places, _ = batch
        # calculate descriptors
        descriptors = self(places)
        return descriptors.detach().cpu()
    
    def validation_epoch_end(self, val_step_outputs):
        """
        at the end of each validation epoch
        descriptors are returned in their order
        depending on how the validation dataset is implemented 
        for this project (MSLS val, Pittburg val), it is always references then queries.
        For example, if we have n references and m queries, we will get 
        the descriptors for each val_dataset in a list as follows: 
        [R1, R2, ..., Rn, Q1, Q2, ..., Qm]
        we then split it to references=[R1, R2, ..., Rn] and queries=[Q1, Q2, ..., Qm]
        to calculate recall@K using the ground truth provided.
        """
        dm = self.trainer.datamodule
        # The following line is a hack: if we have only one validation set, then
        # we need to put the outputs in a list (Pytorch Lightning does not do it presently)
        if len(dm.val_datasets)==1: # we need to put the outputs in a list
            
            # val_step_outputs contains lists of lists for references and queries's descriptors
            # each list rapresent a validation dataset and contains the relative lists/batches of descriptors
            val_step_outputs = [val_step_outputs]

        for i, (val_set_name, val_dataset) in enumerate(zip(dm.val_set_names, dm.val_datasets)):
            # concatenate to have in a single list the descriptors of references and queries of a particular validation set
            feats = torch.concat(val_step_outputs[i], dim=0) 
            
            num_references = val_dataset.num_references
            num_queries = val_dataset.num_queries
            ground_truth = val_dataset.ground_truth
            
            # split to ref and queries    
            r_list = feats[ : num_references]
            q_list = feats[num_references : ]

            recalls_dict, predictions = utils.get_validation_recalls(r_list=r_list, 
                                                q_list=q_list,
                                                k_values=[1, 5, 10, 15, 20, 25],
                                                gt=ground_truth,
                                                print_results=True,
                                                faiss_gpu=self.faiss_gpu, 
                                                dataset_name=val_set_name
                                                )                                 
            del r_list, q_list, feats, num_references, ground_truth

            self.log(f'{val_set_name}/R1', recalls_dict[1], prog_bar=False, logger=True)
            self.log(f'{val_set_name}/R5', recalls_dict[5], prog_bar=False, logger=True)
            self.log(f'{val_set_name}/R10', recalls_dict[10], prog_bar=False, logger=True)
        print('\n\n') 

            
            
if __name__ == '__main__':
    
    pl.utilities.seed.seed_everything(seed=1, workers=True)
    
    # the datamodule contains train and validation dataloaders,
    # refer to ./dataloader/GSVCitiesDataloader.py for details
    # if you want to train on specific cities, you can comment/uncomment
    # cities from the list TRAIN_CITIES
    datamodule = GSVCitiesDataModule(
        batch_size=100, 
        img_per_place=4,
        min_img_per_place=4,
        # cities=['London', 'Boston', 'Melbourne'], # you can specify cities here or in GSVCitiesDataloader.py
        shuffle_all=False, # shuffle all images or keep shuffling in-city only
        random_sample_from_each_place=True,
        image_size=(224, 224),
        num_workers=8,
        show_data_stats=True,
        val_set_names=['sfxs_val'],
    )
    
    model = VPRModel(
        #-------------------------------
        #---- Backbone architecture ----
        backbone_arch='resnet18',
        pretrained=True,
        layers_to_freeze=2,
        layers_to_crop=[4], # 4 crops the last resnet layer, 3 crops the 3rd, ...etc
        
        #---------------------
        #---- Aggregator -----
        
        # agg_arch='GeM',
        # agg_config={'p': 3},
        
        # agg_arch='ConvAP',
        # agg_config={'in_channels': 256,
        #           'out_channels': 1024,
        #           's1' : 2,
        #           's2' : 2},

        # agg_arch='AVG',

        agg_arch='MixVPR',
        agg_config={'in_channels' : 256,
                    'in_h' : 14,
                    'in_w' : 14,
                    'out_channels' : 256,
                    'mix_depth' : 5,
                    'mlp_ratio' : 1,
                    'out_rows' : 4}, # the output dim will be (out_rows * out_channels)


        #-----------------------------------
        #---- Training hyperparameters -----
        #
        lr=0.0005, # 0.0002 for adam
        optimizer='adamw', # adam or adamw
        weight_decay=0.0001, # 0.0 for adam
        momentum=0.9,
        warmpup_steps=600,
        milestones=[5, 10, 15, 25],
        lr_mult=0.3,
        
        #---------------------------------
        #---- Training loss function -----
        # example: ContrastiveLoss, TripletMarginLoss, MultiSimilarityLoss
        #
        loss_name='MultiSimilarityLoss',
        miner_name='MultiSimilarityMiner', # example: TripletMarginMiner, MultiSimilarityMiner, PairMarginMiner
        miner_margin=0.1,
        faiss_gpu=False
    )
    
    # model params saving using Pytorch Lightning
    # we save the best 3 models according to Recall@1 on sfxs val
    checkpoint_cb1 = ModelCheckpoint(
        monitor='sfxs_val/R1',
        filename=f'{model.encoder_arch}' +
        '_epoch({epoch:02d})_step({step:04d})_R1[{sfxs_val/R1:.4f}]_R5[{sfxs_val/R5:.4f}]',
        auto_insert_metric_name=False,
        save_weights_only=True,
        save_top_k=3,
        mode='max'
      )
    
    # saving the most recent training checkpoint each 20 steps/iterations
    checkpoint_cb2 = ModelCheckpoint(
        monitor='step',
        filename=f'{model.encoder_arch}' +
        '_epoch({epoch:02d})_step({step:04d})_training_checkpoint',
        save_top_k=1,
        every_n_train_steps=20,
        mode='max',
      )

    #------------------
    # we instanciate a trainer
    trainer = pl.Trainer(
        accelerator='gpu', devices=[0],
        default_root_dir=f'/content/drive/MyDrive/Project/code/LOGS/{model.encoder_arch}', # Tensorflow can be used to viz 

        num_sanity_val_steps=0, # runs N validation steps before stating training
        precision=16, # we use half precision to reduce  memory usage (and 2x speed on RTX)
        max_epochs=30,
        check_val_every_n_epoch=1, # run validation every epoch
        callbacks=[checkpoint_cb1, checkpoint_cb2], # we run the checkpointing callback (you can add more)
        log_every_n_steps=20,
        reload_dataloaders_every_n_epochs=1, # we reload the dataset to shuffle the order
        # fast_dev_run=True # comment if you want to start training the network and saving checkpoints
    )


    # we call the trainer, and give it the model and the datamodule
    # now you see the modularity of Pytorch Lighning?  
    # the presence of checkpoints is checked and the best one (max epoch and step) is given to the trainer
    # so that the training of the model can continue from where it interrupted

    # recursively retrieving the list of checkpoints names
    checkpoints = glob.glob('**/*_training_checkpoint.ckpt', 
                      root_dir='/content/drive/MyDrive/Project/code/LOGS/resnet18/lightning_logs',
                      recursive=True)
    # no stored run
    if not checkpoints:
      # trainig from start
      trainer.fit(model=model, datamodule=datamodule)
    else:
      # mapping each file name into the tuple (epoch, step)
      epoch_step = [(int(f_name.split('epoch=')[1].split(')')[0]), int(f_name.split('step=')[1].split(')')[0])) for f_name in checkpoints]
      # retrieving the index of the highest one
      best_pair_index = max(enumerate(epoch_step), key=lambda t: t[1])[0]
      # selecting the best checkpoint
      best_ckpt = checkpoints[best_pair_index]
      # training from the best checkpoint
      trainer.fit(model=model, 
                    datamodule=datamodule, 
                    ckpt_path='/content/drive/MyDrive/Project/code/LOGS/resnet18/lightning_logs/'+best_ckpt)
