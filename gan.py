
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from skimage.measure import compare_ssim,compare_psnr,compare_mse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self,x):
        out = self.model(x.view(x.size(0),784))
        out = out.view(out.size(0), -1)
        return out

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(256,momentum=0.8),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(512,momentum=0.8),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(1024,momentum=0.8),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.view(x.size(0), 100)
        out = self.model(x)
        return out



def train_discriminator(criterion,discriminator, images, real_labels, fake_images, fake_labels):
    discriminator.zero_grad()
    outputs_real = discriminator(images)
    real_loss = criterion(outputs_real, real_labels)
    real_score = outputs_real

    real_true_count = 0
    for i in range(real_score.size(0)):
        if real_score[i] >= 0.5:
            real_true_count += 1

    real_disc_accuracy = 100*(real_true_count/real_score.size(0))


    outputs_fake = discriminator(fake_images)
    fake_loss = criterion(outputs_fake, fake_labels)
    fake_score = outputs_fake

    fake_true_count = 0
    for i in range(fake_score.size(0)):
        if fake_score[i] < 0.5:
            fake_true_count += 1

    fake_disc_accuracy = 100*(fake_true_count/fake_score.size(0))



    d_loss = (real_loss + fake_loss)/2.0
    d_loss.backward()
    d_optimizer.step()
    return d_loss, real_score, fake_score, real_disc_accuracy, fake_disc_accuracy

def train_generator(criterion,generator, discriminator_outputs, real_labels):
    generator.zero_grad()
    g_loss = criterion(discriminator_outputs, real_labels)
    g_loss.backward()
    g_optimizer.step()
    return g_loss

def datasets():
    dataset = pd.read_csv('mnist.csv', sep=',').values[:, 1:].reshape(-1, 28, 28)
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(dataset, test_size=0.2)

    X_train = (train - 127.5) / 127.5
    Y_train = train

    X_test = (test - 127.5) / 127.5
    Y_test = test

    return X_train.astype(np.float32), Y_train.astype(np.float32), X_test.astype(np.float32), Y_test.astype(np.float32)




def train_gan(criterion,discriminator,generator,num_epochs,batch_size,save_interval):
    X_train, _, _, _ = datasets()
    print(X_train.shape)
    d_loss_list = []
    g_loss_list = []
    epoch_list = []
    d_real_acc_list = []
    d_fake_acc_list = []
    real_score_list = []
    fake_score_list = []

    for epoch in range(1,num_epochs+1):
        """for batch in range(0, X_train.shape[0], batch_size):
            if X_train.shape[0] - batch < batch_size:
                minibatch_X = torch.from_numpy(X_train[batch:, :, :])
            else:
                minibatch_X = torch.from_numpy(X_train[batch:batch + batch_size, :, :])"""

        idx = np.random.randint(0, X_train.shape[0], batch_size)
        minibatch_X = torch.from_numpy(X_train[idx, :, :])

        images = Variable(minibatch_X.cuda())
        noise = Variable(torch.from_numpy(np.random.normal(0,1,(minibatch_X.size(0),100))).cuda().float())

        real_labels = Variable(torch.ones(minibatch_X.size(0),1).cuda())
        fake_labels = Variable(torch.zeros(minibatch_X.size(0),1).cuda())
        fake_images = generator(noise)

        d_loss, real_score, fake_score, real_d_acc, fake_d_acc = train_discriminator(criterion,discriminator, images, real_labels, fake_images,
                                                                 fake_labels)
        noise = Variable(torch.from_numpy(np.random.normal(0,1,(minibatch_X.size(0),100))).cuda().float())
        fake_images = generator(noise)
        outputs = discriminator(fake_images)

        g_loss = train_generator(criterion, generator, outputs, real_labels)


        print("epoch: " + str(epoch) + "\n" +
              "d_loss: " + str(d_loss.item()) + "\n" +
              "g_loss: " + str(g_loss.item()))
        #print("disc real acc: "+str(real_d_acc))
        #print("disc fake acc: "+str(fake_d_acc))
        print("-----------------")

        if epoch % save_interval == 0 or epoch == 1:
            d_loss_list.append(d_loss.item())
            g_loss_list.append(g_loss.item())
            d_real_acc_list.append(real_d_acc)
            d_fake_acc_list.append(fake_d_acc)
            epoch_list.append(epoch)
            real_score_list.append(real_score.cpu().detach().numpy().astype(np.float32))
            fake_score_list.append(fake_score.cpu().detach().numpy().astype(np.float32))
            sample_images(generator,epoch)


    return epoch_list,d_loss_list,g_loss_list,d_real_acc_list,d_fake_acc_list,np.array(real_score_list),np.array(fake_score_list)

def sample_images(generator,epoch):
    r, c = 5, 5
    noise = Variable(torch.from_numpy(np.random.normal(0,1,(r*c,100))).cuda().float())
    gen_imgs = generator(noise)
    #print(gen_imgs)
    # Rescale images 0 - 1
    gen_imgs = 127.5 * gen_imgs + 127.5
    gen_imgs = gen_imgs.cpu().detach().numpy().astype(np.float32).reshape(-1,28,28)

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[cnt, :, :], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig("images/%d.png" % epoch)
    plt.close()

if __name__ == '__main__':
    discriminator = Discriminator().cuda()
    generator = Generator().cuda()

    criterion = nn.BCELoss()
    lr = 0.0002
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999), eps=1.e-7)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999), eps=1.e-7)

    epoch,d_loss,g_loss,d_r_acc,d_f_acc,real_predictions,fake_predictions = train_gan(criterion=criterion,
                                                                                    discriminator=discriminator,
                                                                                    generator=generator,
                                                                                    num_epochs=30000,
                                                                                    batch_size=128,
                                                                                    save_interval=200)

    plt.plot(epoch, d_loss, 'b', label="discriminator loss")
    plt.plot(epoch, g_loss, 'r', label="generator loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.legend()

    plt.plot(epoch, list((np.array(d_r_acc)+np.array(d_f_acc))/2), 'b', label="discriminator acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.legend()

    accuracy = (np.array(d_r_acc)+np.array(d_f_acc))/2

    np.savetxt("discriminator_real_predictions.txt",real_predictions.reshape(-1,1))
    np.savetxt("discriminator_fake_predictions.txt",fake_predictions.reshape(-1,1))
    np.savetxt("discriminator_accuracy.txt",accuracy.reshape(-1,1))
    np.savetxt("discriminator_loss.txt",np.array(d_loss).reshape(-1,1))
    np.savetxt("generator_loss.txt",np.array(g_loss).reshape(-1,1))