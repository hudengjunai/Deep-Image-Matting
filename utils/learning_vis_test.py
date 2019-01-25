from data import get_train_val_dataloader
from .visulization import Visulizer


vis = Visulizer('env'=main)
vis.log('the is a learning precodure visulizaiton')

train_loader,valid_loader = get_train_val_dataloader(batch_size=4,num_workers=3)
for i,(data,label) in enumerate(train_loader):
    vis.